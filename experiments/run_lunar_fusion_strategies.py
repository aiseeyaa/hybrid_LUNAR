"""
CO TU ROBIMY (BEZ WLASNEGO TUNINGU LUNAR-a/IF):
STATUS: eksperyment archiwalny/opcjonalny - w duzej mierze zdublowany przez
eksperyment 4 (run_lunar_with_best_baseline.py) i eksperyment 7
(run_robustness_analysis.py). Zostaw jako dodatkowy, jesli akurat interesuje
Cie konkretnie para LUNAR+IsolationForest (nie automatycznie wybierany
najlepszy baseline).

ZMIANA: hiperparametry LUNAR-a i IsolationForest wczytujemy z ich solo-
wynikow (run_single_experiment.py, run_isolation_forest.py) zamiast tunowac
je od nowa. Jedyne tunowanie w tym skrypcie to dobor strategii fuzji
(tune_meta_fusion) dla kazdego punktu siatki noise/imbalance.

POPRAWKA (z wczesniejszej wersji): niezbalansowanie stosujemy na ZBIORZE
TESTOWYM, nie na train (train jest z zalozenia czysto normalny - patrz
utils.load_dataset).

WYMAGA wczesniej uruchomionych: run_single_experiment.py,
run_isolation_forest.py dla tego samego (dataset, run_index).

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

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
EXTERNAL_LUNAR_DIR = ROOT / "external" / "LUNAR"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

sys.path.append(str(SRC_DIR))
sys.path.append(str(EXTERNAL_LUNAR_DIR))

from data import make_final_subsample, clear_dataset_cache
from metrics import minmax_scale_scores, print_metrics
from results import build_experiment_record, save_record_json
from threshold_reporting import calibrate_and_evaluate
from ensemble_utils import tune_meta_fusion, apply_meta_fusion
from baseline_selection import load_baseline_params, BASELINE_REGISTRY
from lunar_params_loading import load_lunar_params

import LUNAR
import variables as var

SEED = 29
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
NOISE_LEVELS = [0.0, 0.05, 0.15]
IMBALANCE_FACTORS = [1.0, 0.5, 0.25]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "LUNAR_Fusion_v2"

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


def add_noise(x, level, seed):
    if level <= 0:
        return x
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, level, size=x.shape)


def downsample_anomalies(x, y, factor, seed):
    if factor >= 1.0:
        return x, y
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        raise ValueError(
            "downsample_anomalies: przekazany zbior nie ma zadnych probek anomalii (y==1). "
            "Niezbalansowanie nalezy symulowac na val_calib/test, nie na train."
        )
    neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    keep_pos = rng.choice(pos_idx, size=max(1, int(len(pos_idx) * factor)), replace=False)
    keep_idx = np.concatenate([neg_idx, keep_pos])
    rng.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]


def run_experiment(dataset, run_cfg, lunar_params, if_params):
    run_index = run_cfg["run_index"]

    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=lunar_params["k"],
    )

    variant_records = []
    for noise in NOISE_LEVELS:
        x_train_mod = add_noise(train_x, noise, SEED)

        for imb in IMBALANCE_FACTORS:
            x_test_mod, y_test_mod = downsample_anomalies(test_x, test_y, imb, SEED)

            lunar_val, lunar_test, tr_lunar, inf_lunar = score_lunar(
                lunar_params, dataset, x_train_mod, train_y, val_x, val_y, x_test_mod, y_test_mod
            )
            if_val, if_test, tr_if, inf_if = BASELINE_REGISTRY["IsolationForest"]["scorer"](
                if_params, x_train_mod, val_x, x_test_mod, SEED
            )

            val_matrix = np.column_stack([lunar_val, if_val])
            test_matrix = np.column_stack([lunar_test, if_test])
            best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
            fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)

            metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
                val_y, fused_val, y_test_mod, fused_test,
                beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
            )
            print_metrics(f"{MODEL_TYPE} (standard) - {dataset} run{run_index} noise={noise} imb_test={imb}", metrics)

            variant_records.append({
                "noise": noise, "imbalance_factor_test": imb, "metrics": metrics,
                "runtime_train": tr_lunar + tr_if, "runtime_inference": inf_lunar + inf_if,
                "threshold_info": threshold_info, "threshold_variants": threshold_variants, "pr_curve": pr_curve,
                "fusion_strategy": best_meta["fusion_strategy"], "meta": best_meta,
            })
            del val_matrix, test_matrix, fused_val, fused_test
            cleanup_memory()

    del train_x, train_y, val_x, val_y, test_x, test_y
    cleanup_memory()
    clear_dataset_cache()
    return variant_records


def save_results(dataset, run_cfg, lunar_params, if_params, variant_records):
    run_index = run_cfg["run_index"]
    saved = []
    for variant in variant_records:
        record = build_experiment_record(
            dataset_name=dataset, dataset_version=DATASET_VERSION, split_method=SPLIT_METHOD, seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION, model_type=MODEL_TYPE, fusion_strategy=variant["fusion_strategy"],
            hyperparameters={
                "lunar": lunar_params, "if": if_params, "meta": variant["meta"],
                "noise": variant["noise"], "imbalance_factor_test": variant["imbalance_factor_test"],
            },
            metrics=variant["metrics"], runtime_train=variant["runtime_train"], runtime_inference=variant["runtime_inference"],
            threshold_info=variant["threshold_info"], threshold_variants=variant["threshold_variants"], pr_curve=variant["pr_curve"],
        )
        tag = f"{MODEL_TYPE}_noise{variant['noise']}_imbtest{variant['imbalance_factor_test']}"
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
    if_params = load_baseline_params(RESULTS_DIR, args.run_index, "IsolationForest", args.dataset)
    variant_records = run_experiment(args.dataset, run_cfg, lunar_params, if_params)
    saved = save_results(args.dataset, run_cfg, lunar_params, if_params, variant_records)

    best_f1 = max(r["F1"] for r in saved)
    print(f"Finished {args.dataset} run {args.run_index}: {len(saved)} variants, best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
