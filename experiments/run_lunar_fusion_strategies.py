import sys
import gc
import argparse
import time
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

from data import make_optuna_subsample, make_final_subsample, clear_dataset_cache
from optuna_utils import run_study
from metrics import minmax_scale_scores, evaluate_scores, print_metrics, compare_threshold_strategies
from results import build_experiment_record, save_record_json
from ensemble_utils import tune_if, score_if, tune_meta_fusion, apply_meta_fusion

import LUNAR
import variables as var
from sklearn.metrics import roc_auc_score

SEED = 29
N_TRIALS = 200
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
SAMPLE_TYPES = ["UNIFORM", "SUBSPACE", "MIXED"]
NOISE_LEVELS = [0.0, 0.05, 0.15]
IMBALANCE_FACTORS = [1.0, 0.5, 0.25]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_test_fixed_seed"
MODEL_TYPE = "LUNAR_Fusion"

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
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def tune_lunar(train_x, train_y, val_x, val_y, dataset, seed, n_trials, results_dir):
    def objective(trial):
        n_pos = int(val_y.sum())
        n_neg = len(val_y) - n_pos
        if n_pos < 5 or n_neg < 5:
            raise optuna.exceptions.TrialPruned()

        k = trial.suggest_int("k", 5, 150, log=True)
        samples = trial.suggest_categorical("samples", SAMPLE_TYPES)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-4, 1.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.5)
        proportion = trial.suggest_int("proportion", 1, 2)
        n_epochs = trial.suggest_int("n_epochs", 50, 300, step=25)

        var.lr, var.wd, var.epsilon = lr, wd, epsilon
        var.proportion, var.n_epochs = proportion, n_epochs

        try:
            out = LUNAR.run(
                train_x, train_y, val_x, val_y, val_x, val_y,
                dataset, seed, k, samples, train_new_model=True,
            )
            auc = roc_auc_score(val_y, out.numpy())
        except RuntimeError as e:
            if "memory" in str(e).lower():
                cleanup_memory()
                raise optuna.exceptions.TrialPruned()
            raise
        except ValueError:
            raise optuna.exceptions.TrialPruned()

        return auc

    return run_study(objective, f"LUNAR_fusion_{dataset}", seed, n_trials, results_dir=results_dir).best_params


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
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_pos = rng.choice(pos_idx, size=max(1, int(len(pos_idx) * factor)), replace=False)
    keep_idx = np.concatenate([neg_idx, keep_pos])
    rng.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]


def run_optuna(dataset, run_cfg):
    opt_train_x, opt_train_y, opt_val_x, opt_val_y = make_optuna_subsample(
        dataset, SEED, run_cfg["n_train_opt"], run_cfg["n_val_opt"]
    )
    lunar_params = tune_lunar(opt_train_x, opt_train_y, opt_val_x, opt_val_y, dataset, SEED, N_TRIALS, RESULTS_DIR)
    if_params = tune_if(opt_train_x, opt_val_x, opt_val_y, SEED, N_TRIALS, RESULTS_DIR)

    del opt_train_x, opt_train_y, opt_val_x, opt_val_y
    cleanup_memory()
    clear_dataset_cache()
    return {"lunar": lunar_params, "if": if_params}


def run_experiment(dataset, run_cfg, best_params):
    run_index = run_cfg["run_index"]
    lunar_params = best_params["lunar"]
    if_params = best_params["if"]

    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED,
        run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=lunar_params["k"],
    )

    variant_records = []
    for noise in NOISE_LEVELS:
        for imb in IMBALANCE_FACTORS:
            x_train_mod = add_noise(train_x, noise, SEED)
            x_train_mod, y_train_mod = downsample_anomalies(x_train_mod, train_y, imb, SEED)

            lunar_val, lunar_test, tr_lunar, inf_lunar = score_lunar(
                lunar_params, dataset, x_train_mod, y_train_mod, val_x, val_y, test_x, test_y
            )
            if_val, if_test, tr_if, inf_if = score_if(if_params, x_train_mod, val_x, test_x, SEED)

            val_matrix = np.column_stack([lunar_val, if_val])
            test_matrix = np.column_stack([lunar_test, if_test])

            best_meta = tune_meta_fusion(val_matrix, val_y, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
            fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, SEED)

            threshold_candidates = compare_threshold_strategies(val_y, fused_val, beta=2.0, normal_q=0.99)
            chosen_threshold_info = {"selection_method": "validation_f1_max", **threshold_candidates["f1_max"]}
            best_threshold = chosen_threshold_info["threshold"]

            metrics = evaluate_scores(test_y, fused_test, threshold=best_threshold)
            print_metrics(f"{MODEL_TYPE} final - {dataset} run{run_index} noise={noise} imb={imb}", metrics)

            variant_records.append({
                "noise": noise,
                "imbalance_factor": imb,
                "metrics": metrics,
                "runtime_train": tr_lunar + tr_if,
                "runtime_inference": inf_lunar + inf_if,
                "threshold_info": {"candidates": threshold_candidates, "chosen": chosen_threshold_info},
                "fusion_strategy": best_meta["fusion_strategy"],
                "meta": best_meta,
            })

            del val_matrix, test_matrix, fused_val, fused_test
            cleanup_memory()

    del train_x, train_y, val_x, val_y, test_x, test_y
    cleanup_memory()
    clear_dataset_cache()
    return variant_records


def save_results(dataset, run_cfg, best_params, variant_records):
    run_index = run_cfg["run_index"]
    saved = []
    for variant in variant_records:
        record = build_experiment_record(
            dataset_name=dataset,
            dataset_version=DATASET_VERSION,
            split_method=SPLIT_METHOD,
            seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION,
            model_type=MODEL_TYPE,
            fusion_strategy=variant["fusion_strategy"],
            hyperparameters={
                "lunar": best_params["lunar"], "if": best_params["if"], "meta": variant["meta"],
                "noise": variant["noise"], "imbalance_factor": variant["imbalance_factor"],
            },
            metrics=variant["metrics"],
            runtime_train=variant["runtime_train"],
            runtime_inference=variant["runtime_inference"],
            threshold_info=variant["threshold_info"],
        )
        tag = f"{MODEL_TYPE}_noise{variant['noise']}_imb{variant['imbalance_factor']}"
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
    best_params = run_optuna(args.dataset, run_cfg)
    variant_records = run_experiment(args.dataset, run_cfg, best_params)
    saved = save_results(args.dataset, run_cfg, best_params, variant_records)

    best_f1 = max(r["F1"] for r in saved)
    print(f"Finished {args.dataset} run {args.run_index}: {len(saved)} variants, best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
