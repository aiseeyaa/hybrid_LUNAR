import sys
import gc
import argparse
from pathlib import Path

import joblib
import optuna

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"

sys.path.append(str(SRC_DIR))

from data import make_optuna_subsample, make_final_subsample, clear_dataset_cache
from optuna_utils import run_study
from metrics import minmax_scale_scores, evaluate_scores, print_metrics, compare_threshold_strategies
from results import build_experiment_record, save_record_json

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import time

SEED = 81
RESULTS_DIR = ROOT / "results" / f"seed_{SEED}"   
MODELS_DIR = ROOT / "models" / f"seed_{SEED}" 
N_TRIALS = 200
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_test_fixed_seed"
MODEL_TYPE = "IsolationForest"
FUSION_STRATEGY = "none"

RUN_CONFIGS = {
        1: dict(
            run_index=1,
            n_train_opt=7000, n_val_opt=3000,
            n_train_final=40000, n_val_final=12000, n_test_final=80000,
            notes="run1_small_opt_sample",
        ),
        2: dict(
            run_index=2,
            n_train_opt=14000, n_val_opt=6000,
            n_train_final=40000, n_val_final=12000, n_test_final=80000,
            notes="run2_medium_opt_sample",
        ),
        3: dict(
            run_index=3,
            n_train_opt=21000, n_val_opt=9000,
            n_train_final=40000, n_val_final=12000, n_test_final=80000,
            notes="run3_large_opt_sample",
        ),
    }



def cleanup_memory():
    gc.collect()


def make_objective(train_x, val_x, val_y):
    def objective(trial):
        n_pos = int(val_y.sum())
        n_neg = len(val_y) - n_pos
        if n_pos < 5 or n_neg < 5:
            raise optuna.exceptions.TrialPruned()

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "contamination": trial.suggest_float("contamination", 0.01, 0.5),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        try:
            clf = IsolationForest(**params, random_state=SEED, n_jobs=-1)
            clf.fit(train_x)
            val_scores = -clf.score_samples(val_x)
            auc = roc_auc_score(val_y, val_scores)
        except ValueError:
            raise optuna.exceptions.TrialPruned()
        return auc
    return objective


def fit_and_score_if(params, train_x, val_x, test_x):
    clf = IsolationForest(**params, random_state=SEED, n_jobs=-1)

    start_train = time.time()
    clf.fit(train_x)
    runtime_train = time.time() - start_train

    scores_val = minmax_scale_scores(-clf.score_samples(val_x))

    start_inference = time.time()
    scores_test = minmax_scale_scores(-clf.score_samples(test_x))
    runtime_inference = time.time() - start_inference

    return clf, scores_val, scores_test, runtime_train, runtime_inference


def run_optuna(dataset, run_cfg):
    run_index = run_cfg["run_index"]
    study_name = f"IF_{dataset}_run{run_index}"

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

    clf, scores_val, scores_test, runtime_train, runtime_inference = fit_and_score_if(
        best_params, train_x, val_x, test_x
    )

    threshold_candidates = compare_threshold_strategies(val_y, scores_val, beta=2.0, normal_q=0.99)
    chosen_threshold_info = {"selection_method": "validation_f1_max", **threshold_candidates["f1_max"]}
    best_threshold = chosen_threshold_info["threshold"]

    print(f"[{dataset} run{run_index}] threshold candidates: {threshold_candidates}")
    print(f"[{dataset} run{run_index}] chosen threshold={best_threshold:.4f} by validation_f1_max")

    metrics = evaluate_scores(test_y, scores_test, threshold=best_threshold)
    print_metrics(f"IsolationForest final - {dataset} run{run_index}", metrics)

    result_bundle = {
        "model": clf,
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
    model_path = model_dir / f"{dataset}_run{run_index}.joblib"
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
