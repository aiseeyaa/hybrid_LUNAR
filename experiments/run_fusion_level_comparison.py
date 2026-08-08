"""
CO TU ROBIMY (Eksperyment 6 z drabiny ablacyjnej, BEZ WLASNEGO TUNINGU):
Bierzemy zwycieska pare LUNAR + najlepszy klasyczny baseline i porownujemy
TRZY POZIOMY FUZJI (score-level, decision-level OR/AND, feature-level), a
dodatkowo dokladamy jako punkt odniesienia najlepszy wynik self-ensemble
LUNAR-a (eksperyment 3).

ZMIANA: hiperparametry LUNAR-a wczytujemy z run_single_experiment.py, a
hiperparametry klasycznego baseline'u z jego wlasnego solo-wyniku - nic tu
nie jest tunowane od nowa poza doborem strategii fuzji score-level
(tune_meta_fusion), co jest jedynym nowym elementem tego eksperymentu.

WYMAGA wczesniej uruchomionych:
  - run_single_experiment.py
  - run_isolation_forest.py / run_lof.py / run_ocsvm.py / run_dbscan.py
  - run_lunar_self_ensemble.py
dla tego samego (dataset, run_index).

Nie modyfikuje LUNAR.py, utils.py ani variables.py.
"""

import sys
import gc
import time
import argparse
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.linear_model import LogisticRegression

from data import make_final_subsample, clear_dataset_cache
from metrics import minmax_scale_scores, print_metrics, find_best_f1_threshold
from results import build_experiment_record, save_record_json
from threshold_reporting import calibrate_and_evaluate
from ensemble_utils import tune_meta_fusion, apply_meta_fusion
from baseline_selection import select_best_baseline, BASELINE_REGISTRY
from self_ensemble_selection import select_best_self_ensemble
from lunar_params_loading import load_lunar_params

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
EXTERNAL_LUNAR_DIR = ROOT / "external" / "LUNAR"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

sys.path.append(str(SRC_DIR))
sys.path.append(str(EXTERNAL_LUNAR_DIR))

import LUNAR
import variables as var

SEED = 29
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "Fusion_Level_Comparison"

MIN_RECALL_TARGET = 0.90
FBETA = 2.0
NORMAL_Q = 0.99

RUN_CONFIGS = {
    1: dict(run_index=1, n_train_final=154000, n_val_final=66000, n_test_final=100000, notes="run1"),
    2: dict(run_index=2, n_train_final=154000, n_val_final=66000, n_test_final=100000, notes="run2"),
    3: dict(run_index=3, n_train_final=154000, n_val_final=66000, n_test_final=100000, notes="run3"),
}


def cleanup_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def score_lunar(params, dataset, train_x, train_y, val_x, val_y, test_x, test_y):
    var.lr = params["lr"]; var.wd = params["wd"]; var.epsilon = params["epsilon"]
    var.proportion = params["proportion"]; var.n_epochs = params["n_epochs"]

    original_device = var.device
    var.device = torch.device("cpu")
    try:
        t0 = time.time()
        out_val = LUNAR.run(
            train_x, train_y, val_x, val_y, val_x, val_y,
            dataset, SEED, params["k"], params["samples"], train_new_model=True,
        )
        runtime_train = time.time() - t0
        scores_val = minmax_scale_scores(out_val.numpy())

        cleanup_memory()

        t1 = time.time()
        out_test = LUNAR.run(
            train_x, train_y, val_x, val_y, test_x, test_y,
            dataset, SEED, params["k"], params["samples"], train_new_model=False,
        )
        runtime_inference = time.time() - t1
        scores_test = minmax_scale_scores(out_test.numpy())
        return scores_val, scores_test, runtime_train, runtime_inference
    finally:
        var.device = original_device
        cleanup_memory()


def run_experiment(dataset, run_cfg, lunar_params):
    run_index = run_cfg["run_index"]

    best_model_type, best_record = select_best_baseline(RESULTS_DIR, run_index, dataset)
    best_baseline_params = best_record["hyperparameters"]
    baseline_scorer = BASELINE_REGISTRY[best_model_type]["scorer"]

    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=lunar_params["k"],
    )

    lunar_val, lunar_test, tr_lunar, inf_lunar = score_lunar(
        lunar_params, dataset, train_x, train_y, val_x, val_y, test_x, test_y
    )
    baseline_val, baseline_test, tr_base, inf_base = baseline_scorer(
        best_baseline_params, train_x, val_x, test_x, SEED
    )
    total_train_rt = tr_lunar + tr_base
    total_inf_rt = inf_lunar + inf_base

    variant_results = {}

    val_matrix = np.column_stack([lunar_val, baseline_val])
    test_matrix = np.column_stack([lunar_test, baseline_test])
    best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
    fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)
    variant_results["score_level"] = {
        "scores_val": fused_val, "scores_test": fused_test,
        "detail": {"fusion_strategy": best_meta["fusion_strategy"], "meta": best_meta},
    }

    thr_lunar, *_ = find_best_f1_threshold(val_y, lunar_val)
    thr_base, *_ = find_best_f1_threshold(val_y, baseline_val)
    lunar_val_bin = (lunar_val >= thr_lunar).astype(int)
    base_val_bin = (baseline_val >= thr_base).astype(int)
    lunar_test_bin = (lunar_test >= thr_lunar).astype(int)
    base_test_bin = (baseline_test >= thr_base).astype(int)
    or_val = np.maximum(lunar_val_bin, base_val_bin).astype(float)
    or_test = np.maximum(lunar_test_bin, base_test_bin).astype(float)
    and_val = np.minimum(lunar_val_bin, base_val_bin).astype(float)
    and_test = np.minimum(lunar_test_bin, base_test_bin).astype(float)
    variant_results["decision_level_OR"] = {
        "scores_val": or_val, "scores_test": or_test,
        "detail": {"rule": "OR", "thr_lunar": float(thr_lunar), "thr_baseline": float(thr_base)},
    }
    variant_results["decision_level_AND"] = {
        "scores_val": and_val, "scores_test": and_test,
        "detail": {"rule": "AND", "thr_lunar": float(thr_lunar), "thr_baseline": float(thr_base)},
    }

    val_aug = np.column_stack([val_x, lunar_val, baseline_val])
    test_aug = np.column_stack([test_x, lunar_test, baseline_test])
    feat_clf = LogisticRegression(max_iter=3000, random_state=SEED)
    feat_clf.fit(val_aug, val_y)
    variant_results["feature_level"] = {
        "scores_val": feat_clf.predict_proba(val_aug)[:, 1], "scores_test": feat_clf.predict_proba(test_aug)[:, 1],
        "detail": {"classifier": "LogisticRegression", "note": "scores appended to original features"},
    }

    saved_variants = {}
    for level_name, payload in variant_results.items():
        metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
            val_y, payload["scores_val"], test_y, payload["scores_test"],
            beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
        )
        print_metrics(f"{MODEL_TYPE} [{level_name}] - {dataset} run{run_index} (LUNAR+{best_model_type})", metrics)
        saved_variants[level_name] = {
            "metrics": metrics, "threshold_info": threshold_info, "threshold_variants": threshold_variants,
            "pr_curve": pr_curve, "detail": payload["detail"],
        }

    best_self_ensemble_strategy, self_ensemble_record = select_best_self_ensemble(RESULTS_DIR, run_index, dataset)
    saved_variants["lunar_self_ensemble"] = {
        "metrics": {
            "AUC_ROC": self_ensemble_record["AUC_ROC"], "AUC_PR": self_ensemble_record["AUC_PR"],
            "Precision": self_ensemble_record["Precision"], "Recall": self_ensemble_record["Recall"],
            "F1": self_ensemble_record["F1"], "ConfusionMatrix": self_ensemble_record["ConfusionMatrix"],
            "threshold": self_ensemble_record["threshold"],
            "Recall_at_FPR_0.01": self_ensemble_record.get("Recall_at_FPR_0.01"),
            "Precision_at_Recall_0.80": self_ensemble_record.get("Precision_at_Recall_0.80"),
        },
        "threshold_info": self_ensemble_record.get("threshold_info", {}),
        "threshold_variants": self_ensemble_record.get("threshold_variants", {}),
        "pr_curve": self_ensemble_record.get("pr_curve", []),
        "detail": {"source": "run_lunar_self_ensemble.py", "best_self_ensemble_strategy": best_self_ensemble_strategy},
    }
    print_metrics(
        f"{MODEL_TYPE} [lunar_self_ensemble={best_self_ensemble_strategy}] (REFERENCJA) - {dataset} run{run_index}",
        saved_variants["lunar_self_ensemble"]["metrics"],
    )

    del train_x, train_y, val_x, val_y, test_x, test_y, val_matrix, test_matrix, val_aug, test_aug
    cleanup_memory()
    clear_dataset_cache()

    return {
        "variants": saved_variants, "runtime_train": total_train_rt, "runtime_inference": total_inf_rt,
        "best_baseline_model_type": best_model_type, "best_baseline_params": best_baseline_params,
        "best_self_ensemble_strategy": best_self_ensemble_strategy,
    }


def save_results(dataset, run_cfg, lunar_params, result_bundle):
    run_index = run_cfg["run_index"]
    saved = []
    for level_name, variant in result_bundle["variants"].items():
        is_reference = level_name == "lunar_self_ensemble"
        record = build_experiment_record(
            dataset_name=dataset, dataset_version=DATASET_VERSION, split_method=SPLIT_METHOD, seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION, model_type=MODEL_TYPE, fusion_strategy=level_name,
            hyperparameters=({
                "lunar": lunar_params,
                "best_baseline_model_type": result_bundle["best_baseline_model_type"],
                "best_baseline_params": result_bundle["best_baseline_params"],
                "detail": variant["detail"],
            } if not is_reference else {
                "note": "wartosci wczytane 1:1 z run_lunar_self_ensemble.py, nie liczone od nowa",
                "best_self_ensemble_strategy": result_bundle["best_self_ensemble_strategy"],
                "detail": variant["detail"],
            }),
            metrics=variant["metrics"],
            runtime_train=0.0 if is_reference else result_bundle["runtime_train"],
            runtime_inference=0.0 if is_reference else result_bundle["runtime_inference"],
            threshold_info=variant["threshold_info"], threshold_variants=variant["threshold_variants"],
            pr_curve=variant["pr_curve"],
        )
        tag = f"{MODEL_TYPE}_{level_name}"
        save_record_json(record, RESULTS_DIR, run_index, tag, dataset)
        saved.append(record)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["CICIDS", "UNSW_NB15"])
    parser.add_argument("run_index", type=int, choices=sorted(RUN_CONFIGS))
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run_cfg = RUN_CONFIGS[args.run_index]
    lunar_params = load_lunar_params(RESULTS_DIR, args.run_index, args.dataset)
    result_bundle = run_experiment(args.dataset, run_cfg, lunar_params)
    saved = save_results(args.dataset, run_cfg, lunar_params, result_bundle)

    best_f1 = max(r["F1"] for r in saved)
    best_name = max(saved, key=lambda r: r["F1"])["fusion_strategy"]
    print(f"Finished {args.dataset} run {args.run_index}: {len(saved)} warianty, best F1={best_f1:.4f} ({best_name})")


if __name__ == "__main__":
    main()
