import sys
import gc
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

sys.path.append(str(SRC_DIR))

from data import make_optuna_subsample, make_final_subsample, clear_dataset_cache
from optuna_utils import run_study
from metrics import minmax_scale_scores, evaluate_scores, print_metrics, compare_threshold_strategies
from results import build_experiment_record, save_record_json

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

SEED = 29
N_TRIALS = 200
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_test_fixed_seed"
MODEL_TYPE = "DBSCAN"
FUSION_STRATEGY = "none"

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

# ---------------------------------------------------------------------------
# NOTE on DBSCAN as an anomaly detector:
# DBSCAN has no native predict/score_samples for unseen points - it only
# assigns cluster labels to the data it was fit on. To score held-out
# val/test points we fit DBSCAN on the training data, keep the "core samples"
# it discovers as a reference set of "normal" density regions, and define the
# anomaly score of any new point as its distance to the nearest core sample
# (via a NearestNeighbors index). Larger distance -> more anomalous.
# If DBSCAN finds zero core samples for a given (eps, min_samples) pair, the
# trial is pruned since no meaningful scoring is possible.
# ---------------------------------------------------------------------------


def cleanup_memory():
    gc.collect()


def fit_dbscan_ref(train_x, eps, min_samples, metric):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    db.fit(train_x)
    if len(db.core_sample_indices_) == 0:
        return None, None
    core = np.asarray(train_x)[db.core_sample_indices_]
    nn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1)
    nn.fit(core)
    return db, nn


def make_objective(train_x, val_x, val_y):
    def objective(trial):
        n_pos = int(val_y.sum())
        n_neg = len(val_y) - n_pos
        if n_pos < 5 or n_neg < 5:
            raise optuna.exceptions.TrialPruned()

        eps = trial.suggest_float("eps", 0.05, 5.0, log=True)
        min_samples = trial.suggest_int("min_samples", 3, 50, log=True)
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan"])

        db, nn = fit_dbscan_ref(train_x, eps, min_samples, metric)
        if nn is None:
            raise optuna.exceptions.TrialPruned()

        try:
            val_scores = nn.kneighbors(val_x, n_neighbors=1)[0].ravel()
            auc = roc_auc_score(val_y, val_scores)
        except ValueError:
            raise optuna.exceptions.TrialPruned()
        return auc
    return objective


def fit_and_score_dbscan(params, train_x, val_x, test_x):
    start_train = time.time()
    db, nn = fit_dbscan_ref(train_x, params["eps"], params["min_samples"], params["metric"])
    runtime_train = time.time() - start_train

    if nn is None:
        raise RuntimeError(
            "Final DBSCAN fit produced zero core samples for the chosen "
            "hyperparameters; cannot score val/test points."
        )

    scores_val = minmax_scale_scores(nn.kneighbors(val_x, n_neighbors=1)[0].ravel())

    start_inference = time.time()
    scores_test = minmax_scale_scores(nn.kneighbors(test_x, n_neighbors=1)[0].ravel())
    runtime_inference = time.time() - start_inference

    return nn, scores_val, scores_test, runtime_train, runtime_inference


def run_optuna(dataset, run_cfg):
    run_index = run_cfg["run_index"]
    study_name = f"DBSCAN_{dataset}_run{run_index}"

    train_x, train_y, val_x, val_y = make_optuna_subsample(
        dataset, SEED, run_cfg["n_train_opt"], run_cfg["n_val_opt"]
    )
    objective = make_objective(train_x, val_x, val_y)
    study = run_study(objective, study_name, SEED, N_TRIALS, results_dir=RESULTS_DIR, save_trials=True)
    best_params = dict(study.best_params)

    del objective, study, train_x, train_y, val_x, val_y
    cleanup_memory()
    clear_dataset_cache()
    return best_params


def run_experiment(dataset, run_cfg, best_params):
    run_index = run_cfg["run_index"]
    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"]
    )

    nn, scores_val, scores_test, runtime_train, runtime_inference = fit_and_score_dbscan(
        best_params, train_x, val_x, test_x
    )

    threshold_candidates = compare_threshold_strategies(val_y, scores_val, beta=2.0, normal_q=0.99)
    chosen_threshold_info = {"selection_method": "validation_f1_max", **threshold_candidates["f1_max"]}
    best_threshold = chosen_threshold_info["threshold"]

    print(f"[{dataset} run{run_index}] threshold candidates: {threshold_candidates}")
    print(f"[{dataset} run{run_index}] chosen threshold={best_threshold:.4f} by validation_f1_max")

    metrics = evaluate_scores(test_y, scores_test, threshold=best_threshold)
    print_metrics(f"DBSCAN final - {dataset} run{run_index}", metrics)

    result_bundle = {
        "model": nn,
        "metrics": metrics,
        "runtime_train": runtime_train,
        "runtime_inference": runtime_inference,
        "threshold_info": {"candidates": threshold_candidates, "chosen": chosen_threshold_info},
    }

    del train_x, train_y, val_x, val_y, test_x, test_y, scores_val, scores_test
    cleanup_memory()
    clear_dataset_cache()
    return result_bundle


def save_results(dataset, run_cfg, best_params, result_bundle):
    run_index = run_cfg["run_index"]
    model_dir = MODELS_DIR / MODEL_TYPE
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset}_run{run_index}_nn.joblib"
    joblib.dump(result_bundle["model"], model_path)

    record = build_experiment_record(
        dataset_name=dataset,
        dataset_version=DATASET_VERSION,
        split_method=SPLIT_METHOD,
        seed=SEED,
        preprocessing_version=PREPROCESSING_VERSION,
        model_type=MODEL_TYPE,
        fusion_strategy=FUSION_STRATEGY,
        hyperparameters=best_params,
        metrics=result_bundle["metrics"],
        runtime_train=result_bundle["runtime_train"],
        runtime_inference=result_bundle["runtime_inference"],
        threshold_info=result_bundle["threshold_info"],
        model_path=model_path,
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
