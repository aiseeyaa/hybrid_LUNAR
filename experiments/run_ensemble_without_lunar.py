#  jeden ensemble z LOF, IF, OneClassSVM BEZ LUNAR-a - przy uzyciu 5 strategii fuzji score-level

import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"


sys.path.append(str(SRC_DIR))

from data import make_final_subsample, clear_dataset_cache
from metrics import print_metrics
from results import build_experiment_record, save_record_json
from threshold_reporting import calibrate_and_evaluate
from baseline_selection import load_all_baseline_params, BASELINE_REGISTRY
from ensemble_utils import tune_meta_fusion, apply_meta_fusion

SEED = 81
RESULTS_DIR = ROOT / "results" / f"seed_{SEED}"   
MODELS_DIR = ROOT / "models" / f"seed_{SEED}" 
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "Ensemble_without_LUNAR"

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


def run_experiment(dataset, run_cfg, best_params):
    run_index = run_cfg["run_index"]
    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"]
    )

    if_val, if_test, tr_if, inf_if = BASELINE_REGISTRY["IsolationForest"]["scorer"](
        best_params["IsolationForest"], train_x, val_x, test_x, SEED)
    lof_val, lof_test, tr_lof, inf_lof = BASELINE_REGISTRY["LOF"]["scorer"](
        best_params["LOF"], train_x, val_x, test_x, SEED)
    ocsvm_val, ocsvm_test, tr_ocsvm, inf_ocsvm = BASELINE_REGISTRY["OneClassSVM"]["scorer"](
        best_params["OneClassSVM"], train_x, val_x, test_x, SEED)

    val_matrix = np.column_stack([if_val, lof_val, ocsvm_val])
    test_matrix = np.column_stack([if_test, lof_test, ocsvm_test])

    best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
    fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)

    metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
        val_y, fused_val, test_y, fused_test,
        beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
    )

    print(f"[{dataset} run{run_index}] threshold candidates (na val_calib): {threshold_candidates}")
    print_metrics(f"{MODEL_TYPE} (standard) - {dataset} run{run_index}", metrics)
    print_metrics(f"{MODEL_TYPE} (recall_oriented) - {dataset} run{run_index}", threshold_variants["recall_oriented"])
    print_metrics(f"{MODEL_TYPE} (operational, min_recall={MIN_RECALL_TARGET}) - {dataset} run{run_index}", threshold_variants["operational"])

    result_bundle = {
        "metrics": metrics,
        "runtime_train": tr_if + tr_lof + tr_ocsvm,
        "runtime_inference": inf_if + inf_lof + inf_ocsvm,
        "threshold_info": threshold_info,
        "threshold_variants": threshold_variants,
        "pr_curve": pr_curve,
        "fusion_strategy": best_meta["fusion_strategy"],
        "meta": best_meta,
    }

    del train_x, train_y, val_x, val_y, test_x, test_y, val_matrix, test_matrix, fused_val, fused_test
    cleanup_memory()
    clear_dataset_cache()
    return result_bundle


def save_results(dataset, run_cfg, best_params, result_bundle):
    run_index = run_cfg["run_index"]
    record = build_experiment_record(
        dataset_name=dataset,
        dataset_version=DATASET_VERSION,
        split_method=SPLIT_METHOD,
        seed=SEED,
        preprocessing_version=PREPROCESSING_VERSION,
        model_type=MODEL_TYPE,
        fusion_strategy=result_bundle["fusion_strategy"],
        hyperparameters={**best_params, "meta": result_bundle["meta"]},
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
    best_params = load_all_baseline_params(RESULTS_DIR, args.run_index, args.dataset)
    result_bundle = run_experiment(args.dataset, run_cfg, best_params)
    record = save_results(args.dataset, run_cfg, best_params, result_bundle)

    print(f"Finished {args.dataset} run {args.run_index}: F1={record['F1']:.4f}, AUC_ROC={record['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
