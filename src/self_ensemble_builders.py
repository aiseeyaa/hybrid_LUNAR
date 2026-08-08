"""
CO TU ROBIMY:
Wspolny modul z 4 strategiami budowania self-ensemble LUNAR-a (bagging,
boosting, voting, stacking) z N niezaleznych instancji tego samego,
dostrojonego LUNAR-a. Wydzielony osobno, zeby ta sama logika mogla byc
uzywana zarowno w pelnej siatce (run_lunar_self_ensemble.py, eksperyment 3),
jak i w parametryzowanym probe (run_robustness_self_ensemble_probe.py) -
bez duplikowania kodu miedzy nimi.

Wszystkie funkcje przyjmuja JUZ PRZETWORZONE dane (np. zaszumione,
okrojone cechy, zmieniona prevalence w tescie) - same z siebie zadnej
transformacji nie robia. Kazda funkcja builder_* zwraca krotke:
(scores_val, scores_test, runtime_train_total, runtime_inference_total, info_dict)

Nie modyfikuje LUNAR.py, utils.py ani variables.py - korzysta z LUNAR.run()
tak jak pozostale skrypty.
"""

import time
import gc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from metrics import minmax_scale_scores, find_best_f1_threshold

import LUNAR
import variables as var


def cleanup_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def bootstrap_sample(train_x, train_y, seed, weights=None):
    rng = np.random.default_rng(seed)
    n = len(train_x)
    idx = rng.choice(n, size=n, replace=True, p=weights)
    return train_x[idx], train_y[idx]


def fit_lunar_instance(params, dataset, instance_seed, train_x, train_y, val_x, val_y, test_x, test_y):
    var.lr = params["lr"]; var.wd = params["wd"]; var.epsilon = params["epsilon"]
    var.proportion = params["proportion"]; var.n_epochs = params["n_epochs"]

    original_device = var.device
    var.device = torch.device("cpu")
    try:
        t0 = time.time()
        out_val = LUNAR.run(
            train_x, train_y, val_x, val_y, val_x, val_y,
            dataset, instance_seed, params["k"], params["samples"], train_new_model=True,
        )
        runtime_train = time.time() - t0
        scores_val = minmax_scale_scores(out_val.numpy())

        cleanup_memory()

        t1 = time.time()
        out_test = LUNAR.run(
            train_x, train_y, val_x, val_y, test_x, test_y,
            dataset, instance_seed, params["k"], params["samples"], train_new_model=False,
        )
        runtime_inference = time.time() - t1
        scores_test = minmax_scale_scores(out_test.numpy())
        return scores_val, scores_test, runtime_train, runtime_inference
    finally:
        var.device = original_device
        cleanup_memory()


def build_bagging(lunar_params, dataset, seed, n_instances, train_x, train_y, val_x, val_y, test_x, test_y):
    """N instancji na roznych bootstrapach train_x/train_y, scory usredniane."""
    val_scores, test_scores = [], []
    total_tr, total_inf = 0.0, 0.0
    for i in range(n_instances):
        instance_seed = seed + i
        boot_x, boot_y = bootstrap_sample(train_x, train_y, instance_seed)
        s_val, s_test, tr, inf = fit_lunar_instance(
            lunar_params, dataset, instance_seed, boot_x, boot_y, val_x, val_y, test_x, test_y
        )
        val_scores.append(s_val); test_scores.append(s_test)
        total_tr += tr; total_inf += inf
        cleanup_memory()
    val_matrix = np.column_stack(val_scores)
    test_matrix = np.column_stack(test_scores)
    return val_matrix.mean(axis=1), test_matrix.mean(axis=1), total_tr, total_inf, {"n_instances": n_instances}


def build_voting(lunar_params, dataset, seed, n_instances, train_x, train_y, val_x, val_y, test_x, test_y):
    """N instancji, kazda binaryzuje wlasnym progiem F1-max na val, finalny
    wynik = udzial glosow 'za anomalia' (fuzja na poziomie DECYZJI)."""
    val_scores, test_scores, thresholds = [], [], []
    total_tr, total_inf = 0.0, 0.0
    for i in range(n_instances):
        instance_seed = seed + i
        boot_x, boot_y = bootstrap_sample(train_x, train_y, instance_seed)
        s_val, s_test, tr, inf = fit_lunar_instance(
            lunar_params, dataset, instance_seed, boot_x, boot_y, val_x, val_y, test_x, test_y
        )
        thr, *_ = find_best_f1_threshold(val_y, s_val)
        val_scores.append(s_val); test_scores.append(s_test); thresholds.append(thr)
        total_tr += tr; total_inf += inf
        cleanup_memory()
    val_votes = np.column_stack([(val_scores[i] >= thresholds[i]).astype(int) for i in range(n_instances)])
    test_votes = np.column_stack([(test_scores[i] >= thresholds[i]).astype(int) for i in range(n_instances)])
    return val_votes.mean(axis=1), test_votes.mean(axis=1), total_tr, total_inf, {
        "n_instances": n_instances, "per_instance_thresholds": [float(t) for t in thresholds]
    }


def build_stacking(lunar_params, dataset, seed, n_instances, train_x, train_y, val_x, val_y, test_x, test_y):
    """N instancji, scory jako cechy do regresji logistycznej (meta-model)."""
    val_scores, test_scores = [], []
    total_tr, total_inf = 0.0, 0.0
    for i in range(n_instances):
        instance_seed = seed + i
        boot_x, boot_y = bootstrap_sample(train_x, train_y, instance_seed)
        s_val, s_test, tr, inf = fit_lunar_instance(
            lunar_params, dataset, instance_seed, boot_x, boot_y, val_x, val_y, test_x, test_y
        )
        val_scores.append(s_val); test_scores.append(s_test)
        total_tr += tr; total_inf += inf
        cleanup_memory()
    val_matrix = np.column_stack(val_scores)
    test_matrix = np.column_stack(test_scores)
    meta = LogisticRegression(max_iter=3000, random_state=seed)
    meta.fit(val_matrix, val_y)
    return (meta.predict_proba(val_matrix)[:, 1], meta.predict_proba(test_matrix)[:, 1],
            total_tr, total_inf, {"n_instances": n_instances, "meta_model": "LogisticRegression"})


def build_boosting(lunar_params, dataset, seed, n_instances, train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Uproszczony AdaBoost: kazda kolejna instancja trenowana na probce
    wazonej wg trudnosci poprzednich rund (na podstawie bledu na val_calib,
    zeby nie dotykac train_y w sposob niezgodny z semi-supervised setupem).
    Finalny wynik = wazona suma scorow (wagi = alpha kazdej rundy).
    """
    weights = np.ones(len(train_x)) / len(train_x)
    val_scores, test_scores, alphas = [], [], []
    total_tr, total_inf = 0.0, 0.0
    for i in range(n_instances):
        instance_seed = seed + i
        boot_x, boot_y = bootstrap_sample(train_x, train_y, instance_seed, weights=weights)
        s_val, s_test, tr, inf = fit_lunar_instance(
            lunar_params, dataset, instance_seed, boot_x, boot_y, val_x, val_y, test_x, test_y
        )
        total_tr += tr; total_inf += inf

        thr, *_ = find_best_f1_threshold(val_y, s_val)
        pred = (s_val >= thr).astype(int)
        err = np.clip(np.average((pred != val_y).astype(float)), 1e-6, 1 - 1e-6)
        alpha = 0.5 * np.log((1 - err) / err)
        alphas.append(alpha)
        val_scores.append(s_val); test_scores.append(s_test)

        weights = weights * np.exp(alpha * (1 if err > 0.3 else -1))
        weights = weights / weights.sum()
        cleanup_memory()

    alphas = np.array(alphas)
    alphas = alphas / max(alphas.sum(), 1e-12)
    val_matrix = np.column_stack(val_scores)
    test_matrix = np.column_stack(test_scores)
    return (val_matrix @ alphas, test_matrix @ alphas, total_tr, total_inf,
            {"n_instances": n_instances, "alphas": [float(a) for a in alphas]})


BUILDERS = {
    "bagging": build_bagging,
    "boosting": build_boosting,
    "voting": build_voting,
    "stacking": build_stacking,
}


def build_self_ensemble(strategy, lunar_params, dataset, seed, n_instances,
                         train_x, train_y, val_x, val_y, test_x, test_y):
    if strategy not in BUILDERS:
        raise ValueError(f"Nieznana strategia self-ensemble: {strategy}. Dostepne: {list(BUILDERS)}")
    return BUILDERS[strategy](lunar_params, dataset, seed, n_instances, train_x, train_y, val_x, val_y, test_x, test_y)
