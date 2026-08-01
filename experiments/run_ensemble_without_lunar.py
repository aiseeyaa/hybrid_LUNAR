import sys
import gc
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"

sys.path.append(str(SRC_DIR))

from data import make_optuna_subsample, make_final_subsample, clear_dataset_cache
from metrics import evaluate_scores, print_metrics, compare_threshold_strategies
from results import build_experiment_record, save_record_json
from ensemble_utils import (
    tune_if, tune_lof, tune_dbscan, tune_ocsvm,
    score_if, score_lof, score_dbscan, score_ocsvm,
    tune_meta_fusion, apply_meta_fusion,
)

import optuna

SEED = 29
N_TRIALS = 200
META_TRIALS = 60
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_test_fixed_seed"
MODEL_TYPE = "Ensemble_without_LUNAR_v2"

BASE_MODELS = ["IF", "LOF", "DBSCAN", "OCSVM"]
TUNERS = {"IF": tune_if, "LOF": tune_lof, "DBSCAN": tune_dbscan, "OCSVM": tune_ocsvm}
SCORERS = {
    "IF": lambda p, train_x, val_x, test_x: score_if(p, train_x, val_x, test_x, SEED),
    "LOF": lambda p, train_x, val_x, test_x: score_lof(p, train_x, val_x, test_x),
    "DBSCAN": lambda p, train_x, val_x, test_x: score_dbscan(p, train_x, val_x, test_x),
    "OCSVM": lambda p, train_x, val_x, test_x: score_ocsvm(p, train_x, val_x, test_x),
}

RUN_CONFIGS = {
    1: dict(run_index=1, n_train_opt=7000, n_val_opt=3000,
            n_train_final=154000, n_val_final=66000, n_test_final=100000,
            notes="run1_small_opt_sample"),
    2: dict(run_index=2, n_train_opt=21000, n_val_opt=9000,
            n_train_final=154000, n_val_final=66000, n_test_final=100000,
            notes="run2_medium_opt_sample"),
    3: dict(run_index=3, n_train_opt=35000, n_val_opt=15000,
            n_train_final=154000, n_val_final=66000, n_test_final=100000,
            notes="run3_large_opt_sample"),
}


def cleanup_memory():
    gc.collect()


def run_optuna(dataset, run_cfg):
    opt_train_x, opt_train_y, opt_val_x, opt_val_y = make_optuna_subsample(
        dataset, SEED, run_cfg["n_train_opt"], run_cfg["n_val_opt"]
    )
    best_params = {
        m: TUNERS[m](opt_train_x, opt_val_x, opt_val_y, SEED, N_TRIALS, RESULTS_DIR)
        for m in BASE_MODELS
    }

    del opt_train_x, opt_train_y, opt_val_x, opt_val_y
    cleanup_memory()
    clear_dataset_cache()
    return best_params


def run_experiment(dataset, run_cfg, best_params):
    run_index = run_cfg["run_index"]
    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"]
    )

    val_cols, test_cols, runtime_train, runtime_inference = [], [], 0.0, 0.0
    for model_name in BASE_MODELS:
        v, t, tr, inf = SCORERS[model_name](best_params[model_name], train_x, val_x, test_x)
        val_cols.append(v); test_cols.append(t)
        runtime_train += tr; runtime_inference += inf

    val_matrix = np.column_stack(val_cols)
    test_matrix = np.column_stack(test_cols)

    best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
    fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)

    threshold_candidates = compare_threshold_strategies(val_y, fused_val, beta=2.0, normal_q=0.99)
    chosen_threshold_info = {"selection_method": "validation_f1_max", **threshold_candidates["f1_max"]}
    best_threshold = chosen_threshold_info["threshold"]

    print(f"[{dataset} run{run_index}] threshold candidates: {threshold_candidates}")
    print(f"[{dataset} run{run_index}] chosen threshold={best_threshold:.4f} by validation_f1_max")

    metrics = evaluate_scores(test_y, fused_test, threshold=best_threshold)
    print_metrics(f"{MODEL_TYPE} final - {dataset} run{run_index}", metrics)

    result_bundle = {
        "metrics": metrics,
        "runtime_train": runtime_train,
        "runtime_inference": runtime_inference,
        "threshold_info": {"candidates": threshold_candidates, "chosen": chosen_threshold_info},
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
        hyperparameters={"base_models": BASE_MODELS, "best_params": best_params, "meta": result_bundle["meta"]},
        metrics=result_bundle["metrics"],
        runtime_train=result_bundle["runtime_train"],
        runtime_inference=result_bundle["runtime_inference"],
        threshold_info=result_bundle["threshold_info"],
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
    best_params = run_optuna(args.dataset, run_cfg)
    result_bundle = run_experiment(args.dataset, run_cfg, best_params)
    record = save_results(args.dataset, run_cfg, best_params, result_bundle)

    print(f"Finished {args.dataset} run {args.run_index}: F1={record['F1']:.4f}, AUC_ROC={record['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
