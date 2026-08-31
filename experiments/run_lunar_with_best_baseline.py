# LUNAR + JEDEN klasyczny model - ten, ktory wypadl najlepiej w eksperymentach solo (02-05).

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

sys.path.append(str(SRC_DIR))
sys.path.append(str(EXTERNAL_LUNAR_DIR))

from data import make_final_subsample, clear_dataset_cache
from metrics import minmax_scale_scores, print_metrics
from results import build_experiment_record, save_record_json
from threshold_reporting import calibrate_and_evaluate
from ensemble_utils import tune_meta_fusion, apply_meta_fusion
from baseline_selection import select_best_baseline, BASELINE_REGISTRY
from lunar_params_loading import load_lunar_params

import LUNAR
import variables as var

SEED = 81
RESULTS_DIR = ROOT / "results" / f"seed_{SEED}"   
MODELS_DIR = ROOT / "models" / f"seed_{SEED}" 
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "LUNAR_plus_BestBaseline"

MIN_RECALL_TARGET = 0.90
FBETA = 2.0
NORMAL_Q = 0.99

RUN_CONFIGS = {
    1: dict(run_index=1, n_train_final=40000, n_val_final=12000, n_test_final=80000, notes="run1"),
    2: dict(run_index=2, n_train_final=40000, n_val_final=12000, n_test_final=80000, notes="run2"),
    3: dict(run_index=3, n_train_final=40000, n_val_final=12000, n_test_final=80000, notes="run3"),
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

    val_matrix = np.column_stack([lunar_val, baseline_val])
    test_matrix = np.column_stack([lunar_test, baseline_test])

    best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
    fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)

    metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
        val_y, fused_val, test_y, fused_test,
        beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
    )

    print(f"[{dataset} run{run_index}] LUNAR + {best_model_type} | threshold candidates: {threshold_candidates}")
    print_metrics(f"{MODEL_TYPE} [LUNAR+{best_model_type}] (standard) - {dataset} run{run_index}", metrics)
    print_metrics(f"{MODEL_TYPE} [LUNAR+{best_model_type}] (recall_oriented) - {dataset} run{run_index}", threshold_variants["recall_oriented"])
    print_metrics(f"{MODEL_TYPE} [LUNAR+{best_model_type}] (operational) - {dataset} run{run_index}", threshold_variants["operational"])

    result_bundle = {
        "metrics": metrics,
        "runtime_train": tr_lunar + tr_base,
        "runtime_inference": inf_lunar + inf_base,
        "threshold_info": threshold_info,
        "threshold_variants": threshold_variants,
        "pr_curve": pr_curve,
        "fusion_strategy": best_meta["fusion_strategy"],
        "meta": best_meta,
        "best_baseline_model_type": best_model_type,
        "best_baseline_params": best_baseline_params,
    }

    del train_x, train_y, val_x, val_y, test_x, test_y, val_matrix, test_matrix, fused_val, fused_test
    cleanup_memory()
    clear_dataset_cache()
    return result_bundle


def save_results(dataset, run_cfg, lunar_params, result_bundle):
    run_index = run_cfg["run_index"]
    record = build_experiment_record(
        dataset_name=dataset,
        dataset_version=DATASET_VERSION,
        split_method=SPLIT_METHOD,
        seed=SEED,
        preprocessing_version=PREPROCESSING_VERSION,
        model_type=MODEL_TYPE,
        fusion_strategy=result_bundle["fusion_strategy"],
        hyperparameters={
            "lunar": lunar_params,
            "best_baseline_model_type": result_bundle["best_baseline_model_type"],
            "best_baseline_params": result_bundle["best_baseline_params"],
            "meta": result_bundle["meta"],
        },
        metrics=result_bundle["metrics"],
        runtime_train=result_bundle["runtime_train"],
        runtime_inference=result_bundle["runtime_inference"],
        threshold_info=result_bundle["threshold_info"],
        threshold_variants=result_bundle["threshold_variants"],
        pr_curve=result_bundle["pr_curve"],
    )
    save_record_json(record, RESULTS_DIR, run_index, MODEL_TYPE, dataset)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["CICIDS", "UNSW_NB15"])
    parser.add_argument("run_index", type=int, choices=sorted(RUN_CONFIGS))
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run_cfg = RUN_CONFIGS[args.run_index]
    lunar_params = load_lunar_params(RESULTS_DIR, args.run_index, args.dataset)
    result_bundle = run_experiment(args.dataset, run_cfg, lunar_params)
    record = save_results(args.dataset, run_cfg, lunar_params, result_bundle)

    print(f"Finished {args.dataset} run {args.run_index}: F1={record['F1']:.4f}, AUC_ROC={record['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
