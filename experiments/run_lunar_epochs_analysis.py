# wrazliwosc LUNAR-a na liczbe epok treningu (siatka 25-300)


import sys
import gc
import time
import argparse
from pathlib import Path

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
from lunar_params_loading import load_lunar_params

import LUNAR
import variables as var

SEED = 29
EPOCH_GRID = [25, 50, 75, 100, 150, 200, 250, 300]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "LUNAR_Epochs_v2"

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


def run_experiment(dataset, run_cfg, base_params):
    run_index = run_cfg["run_index"]
    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=base_params["k"],
    )

    epoch_records = []
    for n_epochs in EPOCH_GRID:
        var.lr = base_params["lr"]; var.wd = base_params["wd"]; var.epsilon = base_params["epsilon"]
        var.proportion = base_params["proportion"]; var.n_epochs = n_epochs

        original_device = var.device
        var.device = torch.device("cpu")
        try:
            t0 = time.time()
            out_val = LUNAR.run(
                train_x, train_y, val_x, val_y, val_x, val_y,
                dataset, SEED, base_params["k"], base_params["samples"], train_new_model=True,
            )
            runtime_train = time.time() - t0
            scores_val = minmax_scale_scores(out_val.numpy())

            cleanup_memory()

            t1 = time.time()
            out_test = LUNAR.run(
                train_x, train_y, val_x, val_y, test_x, test_y,
                dataset, SEED, base_params["k"], base_params["samples"], train_new_model=False,
            )
            runtime_inference = time.time() - t1
            scores_test = minmax_scale_scores(out_test.numpy())
        finally:
            var.device = original_device
            cleanup_memory()

        metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
            val_y, scores_val, test_y, scores_test, beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
        )
        print_metrics(f"{MODEL_TYPE} (standard) - {dataset} run{run_index} n_epochs={n_epochs}", metrics)

        epoch_records.append({
            "n_epochs": n_epochs, "metrics": metrics, "runtime_train": runtime_train, "runtime_inference": runtime_inference,
            "threshold_info": threshold_info, "threshold_variants": threshold_variants, "pr_curve": pr_curve,
        })

    del train_x, train_y, val_x, val_y, test_x, test_y
    cleanup_memory()
    clear_dataset_cache()
    return epoch_records


def save_results(dataset, run_cfg, base_params, epoch_records):
    run_index = run_cfg["run_index"]
    saved = []
    for rec_info in epoch_records:
        record = build_experiment_record(
            dataset_name=dataset, dataset_version=DATASET_VERSION, split_method=SPLIT_METHOD, seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION, model_type=MODEL_TYPE, fusion_strategy="none",
            hyperparameters={**base_params, "n_epochs": rec_info["n_epochs"]},
            metrics=rec_info["metrics"], runtime_train=rec_info["runtime_train"], runtime_inference=rec_info["runtime_inference"],
            threshold_info=rec_info["threshold_info"], threshold_variants=rec_info["threshold_variants"], pr_curve=rec_info["pr_curve"],
        )
        tag = f"{MODEL_TYPE}_{rec_info['n_epochs']}epochs"
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
    base_params = load_lunar_params(RESULTS_DIR, args.run_index, args.dataset)
    epoch_records = run_experiment(args.dataset, run_cfg, base_params)
    saved = save_results(args.dataset, run_cfg, base_params, epoch_records)

    best_f1 = max(r["F1"] for r in saved)
    print(f"Finished {args.dataset} run {args.run_index}: {len(saved)} epoch settings, best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
