"""
Badanie odpornosci dwoch sposobow laczenia LUNAR-a (z najlepszym klasycznym baseline'em ORAZ jako self-ensemble bagging) na trzy czynniki zaklocajace: szum, niezbalansowanie w tescie, feature dropout.
"""

import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
EXTERNAL_LUNAR_DIR = ROOT / "external" / "LUNAR"


sys.path.append(str(SRC_DIR))
sys.path.append(str(EXTERNAL_LUNAR_DIR))

from data import make_final_subsample, clear_dataset_cache
from metrics import print_metrics
from results import build_experiment_record, save_record_json
from threshold_reporting import calibrate_and_evaluate
from ensemble_utils import tune_meta_fusion, apply_meta_fusion
from baseline_selection import select_best_baseline, BASELINE_REGISTRY
from lunar_params_loading import load_lunar_params
from self_ensemble_builders import build_self_ensemble, fit_lunar_instance, cleanup_memory

SEED = 81
RESULTS_DIR = ROOT / "results" / f"seed_{SEED}"   
MODELS_DIR = ROOT / "models" / f"seed_{SEED}" 
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "Robustness_Analysis"

MIN_RECALL_TARGET = 0.90
FBETA = 2.0
NORMAL_Q = 0.99

NOISE_LEVELS = [0.0, 0.05, 0.15]
IMBALANCE_FACTORS = [1.0, 0.5, 0.25]
FEATURE_DROPOUT_LEVELS = [0.0, 0.10, 0.30]
ROBUSTNESS_SELF_ENSEMBLE_N_INSTANCES = 3

RUN_CONFIGS = {
    1: dict(run_index=1, n_train_final=40000, n_val_final=12000, n_test_final=80000, notes="run1"),
    2: dict(run_index=2, n_train_final=40000, n_val_final=12000, n_test_final=80000, notes="run2"),
    3: dict(run_index=3, n_train_final=40000, n_val_final=12000, n_test_final=80000, notes="run3"),
}


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


def feature_dropout(x, level, seed):
    if level <= 0:
        return x
    rng = np.random.default_rng(seed)
    n_features = x.shape[1]
    n_drop = max(1, int(n_features * level))
    drop_idx = rng.choice(n_features, size=n_drop, replace=False)
    x_mod = x.copy()
    x_mod[:, drop_idx] = 0.0
    return x_mod


def _score_pair(lunar_params, baseline_scorer, baseline_params, dataset,
                 x_train, y_train, val_x, val_y, x_test, y_test):
    lunar_val, lunar_test, tr_lunar, inf_lunar = fit_lunar_instance(
        lunar_params, dataset, SEED, x_train, y_train, val_x, val_y, x_test, y_test
    )
    base_val, base_test, tr_base, inf_base = baseline_scorer(baseline_params, x_train, val_x, x_test, SEED)
    val_matrix = np.column_stack([lunar_val, base_val])
    test_matrix = np.column_stack([lunar_test, base_test])
    best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
    fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)
    return fused_val, fused_test, tr_lunar + tr_base, inf_lunar + inf_base, best_meta


def run_experiment(dataset, run_cfg, lunar_params):
    run_index = run_cfg["run_index"]
    best_model_type, best_record = select_best_baseline(RESULTS_DIR, run_index, dataset)
    best_baseline_params = best_record["hyperparameters"]
    baseline_scorer = BASELINE_REGISTRY[best_model_type]["scorer"]

    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=lunar_params["k"],
    )

    all_records = []

    def evaluate_and_collect(factor_name, factor_value, combination_type, fused_val, fused_test,
                              y_val_eval, y_test_eval, tr, inf, fusion_strategy, meta):
        metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
            y_val_eval, fused_val, y_test_eval, fused_test,
            beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
        )
        print_metrics(f"{MODEL_TYPE} [{combination_type}][{factor_name}={factor_value}] - {dataset} run{run_index}", metrics)
        all_records.append({
            "factor_name": factor_name, "factor_value": factor_value, "combination_type": combination_type,
            "metrics": metrics, "runtime_train": tr, "runtime_inference": inf,
            "threshold_info": threshold_info, "threshold_variants": threshold_variants, "pr_curve": pr_curve,
            "fusion_strategy": fusion_strategy, "meta": meta,
        })
        cleanup_memory()

    def run_both_combinations(factor_name, factor_value, x_train_eval, y_train_eval,
                               x_val_eval, y_val_eval, x_test_eval, y_test_eval):
        fused_val, fused_test, tr, inf, best_meta = _score_pair(
            lunar_params, baseline_scorer, best_baseline_params, dataset,
            x_train_eval, y_train_eval, x_val_eval, y_val_eval, x_test_eval, y_test_eval
        )
        evaluate_and_collect(
            factor_name, factor_value, f"lunar_plus_{best_model_type.lower()}",
            fused_val, fused_test, y_val_eval, y_test_eval, tr, inf, best_meta["fusion_strategy"], best_meta
        )

        se_val, se_test, se_tr, se_inf, se_info = build_self_ensemble(
            "bagging", lunar_params, dataset, SEED, ROBUSTNESS_SELF_ENSEMBLE_N_INSTANCES,
            x_train_eval, y_train_eval, x_val_eval, y_val_eval, x_test_eval, y_test_eval
        )
        evaluate_and_collect(
            factor_name, factor_value, "lunar_self_ensemble_bagging",
            se_val, se_test, y_val_eval, y_test_eval, se_tr, se_inf, "bagging",
            {"fusion_strategy": "bagging", "n_instances": ROBUSTNESS_SELF_ENSEMBLE_N_INSTANCES},
        )

    for noise in NOISE_LEVELS:
        x_mod = add_noise(train_x, noise, SEED)
        run_both_combinations("noise", noise, x_mod, train_y, val_x, val_y, test_x, test_y)

    for imb in IMBALANCE_FACTORS:
        x_test_mod, y_test_mod = downsample_anomalies(test_x, test_y, imb, SEED)
        run_both_combinations("imbalance_factor_test", imb, train_x, train_y, val_x, val_y, x_test_mod, y_test_mod)

    for drop_level in FEATURE_DROPOUT_LEVELS:
        x_mod = feature_dropout(train_x, drop_level, SEED)
        val_mod = feature_dropout(val_x, drop_level, SEED)
        test_mod = feature_dropout(test_x, drop_level, SEED)
        run_both_combinations("feature_dropout", drop_level, x_mod, train_y, val_mod, val_y, test_mod, test_y)

    del train_x, train_y, val_x, val_y, test_x, test_y
    cleanup_memory()
    clear_dataset_cache()
    return all_records, best_model_type, best_baseline_params


def save_results(dataset, run_cfg, lunar_params, all_records, best_model_type, best_baseline_params):
    run_index = run_cfg["run_index"]
    saved = []
    for rec_info in all_records:
        record = build_experiment_record(
            dataset_name=dataset, dataset_version=DATASET_VERSION, split_method=SPLIT_METHOD, seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION, model_type=MODEL_TYPE, fusion_strategy=rec_info["fusion_strategy"],
            hyperparameters={
                "lunar": lunar_params, "combination_type": rec_info["combination_type"],
                "best_baseline_model_type": best_model_type, "best_baseline_params": best_baseline_params,
                "meta": rec_info["meta"], rec_info["factor_name"]: rec_info["factor_value"],
            },
            metrics=rec_info["metrics"], runtime_train=rec_info["runtime_train"], runtime_inference=rec_info["runtime_inference"],
            threshold_info=rec_info["threshold_info"], threshold_variants=rec_info["threshold_variants"], pr_curve=rec_info["pr_curve"],
        )
        tag = f"{MODEL_TYPE}_{rec_info['combination_type']}_{rec_info['factor_name']}_{rec_info['factor_value']}"
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
    all_records, best_model_type, best_baseline_params = run_experiment(args.dataset, run_cfg, lunar_params)
    saved = save_results(args.dataset, run_cfg, lunar_params, all_records, best_model_type, best_baseline_params)

    best_f1 = max(r["F1"] for r in saved)
    print(f"Finished {args.dataset} run {args.run_index}: {len(saved)} wariantow, best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
