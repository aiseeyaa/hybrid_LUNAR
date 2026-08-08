"""
CO TU ROBIMY:
Wspolny modul pomocniczy uzywany przez eksperymenty laczace LUNAR-a z
klasycznymi modelami (ensemble, LUNAR+best baseline, porownanie poziomow
fuzji, analiza odpornosci). Zamiast tunowac IsolationForest / LOF /
OneClassSVM / DBSCAN OD NOWA w kazdym skrypcie, WCZYTUJE juz zapisane wyniki
solo-baseline'ow (eksperymenty 02-05) - tak samo jak lunar_params_loading.py
robi to dla LUNAR-a.

Wymaga, zeby run_isolation_forest.py / run_lof.py / run_ocsvm.py /
run_dbscan.py zostaly juz uruchomione dla danego (dataset, run_index) -
w przeciwnym razie rzuca czytelny blad z instrukcja co odpalic.
"""

import json
from pathlib import Path

from results import ordinal_label
from ensemble_utils import score_if, score_lof, score_ocsvm, score_dbscan

BASELINE_MODEL_TYPES = ["IsolationForest", "LOF", "OneClassSVM", "DBSCAN"]

# Ujednolicony rejestr: scorer ma zawsze sygnature
# (params, train_x, val_x, test_x, seed) -> (scores_val, scores_test, runtime_train, runtime_inference)
# nawet jesli dany model nie potrzebuje seeda (np. LOF, OCSVM, DBSCAN) - to
# ujednolica wywolania we wszystkich skryptach.
BASELINE_REGISTRY = {
    "IsolationForest": {
        "scorer": lambda params, train_x, val_x, test_x, seed: score_if(params, train_x, val_x, test_x, seed),
    },
    "LOF": {
        "scorer": lambda params, train_x, val_x, test_x, seed: score_lof(params, train_x, val_x, test_x),
    },
    "OneClassSVM": {
        "scorer": lambda params, train_x, val_x, test_x, seed: score_ocsvm(params, train_x, val_x, test_x),
    },
    "DBSCAN": {
        "scorer": lambda params, train_x, val_x, test_x, seed: score_dbscan(params, train_x, val_x, test_x),
    },
}


def load_baseline_record(results_dir, run_index, model_type, dataset):
    path = Path(results_dir) / f"{ordinal_label(run_index)}_{model_type}_{dataset}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_params(results_dir, run_index, model_type, dataset):
    """Zwraca hiperparametry JUZ dostrojonego klasycznego modelu (bez ponownego tuningu)."""
    rec = load_baseline_record(results_dir, run_index, model_type, dataset)
    if rec is None:
        raise FileNotFoundError(
            f"Brak wynikow solo {model_type} dla dataset={dataset}, run_index={run_index}. "
            f"Najpierw uruchom odpowiedni skrypt solo (run_isolation_forest.py / run_lof.py / "
            f"run_ocsvm.py / run_dbscan.py) dla tego datasetu i run_index."
        )
    return rec["hyperparameters"]


def load_all_baseline_params(results_dir, run_index, dataset):
    """Zwraca {model_type: hiperparametry} dla wszystkich 4 klasycznych modeli naraz."""
    return {mt: load_baseline_params(results_dir, run_index, mt, dataset) for mt in BASELINE_MODEL_TYPES}


def select_best_baseline(results_dir, run_index, dataset, metric="F1"):
    """
    Zwraca (model_type, record) dla klasycznego baseline'u o najwyzszym
    `metric` (domyslnie F1 na zamrozonym zbiorze testowym) dla danego
    (dataset, run_index).
    """
    found = {}
    for model_type in BASELINE_MODEL_TYPES:
        rec = load_baseline_record(results_dir, run_index, model_type, dataset)
        if rec is not None:
            found[model_type] = rec

    if not found:
        raise FileNotFoundError(
            f"Brak wynikow solo-baseline dla dataset={dataset}, run_index={run_index}. "
            f"Najpierw uruchom: run_isolation_forest.py, run_lof.py, run_ocsvm.py, "
            f"run_dbscan.py dla tego datasetu i run_index."
        )

    best_model_type = max(found, key=lambda mt: found[mt][metric])
    print(f"[baseline_selection] najlepszy klasyczny baseline dla {dataset} run{run_index}: "
          f"{best_model_type} ({metric}={found[best_model_type][metric]:.4f})")
    return best_model_type, found[best_model_type]
