import sys
import gc
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

from data import make_optuna_subsample, make_final_subsample, clear_dataset_cache
from optuna_utils import run_study
from metrics import minmax_scale_scores, evaluate_scores, print_metrics, compare_threshold_strategies
from results import build_experiment_record, save_record_json

import LUNAR
import variables as var
import shutil
from sklearn.metrics import roc_auc_score

SEED = 29
N_TRIALS = 200
SAMPLE_TYPES = ["UNIFORM", "SUBSPACE", "MIXED"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_test_fixed_seed"
MODEL_TYPE = "LUNAR"
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


def cleanup_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def get_lunar_internal_model_path(dataset, seed, k):
    return ROOT / "experiments" / "saved_models" / dataset / str(k) / f"net_{seed}.pth"


def copy_lunar_model_to_models_dir(dataset, run_index, seed, k):
    source_path = get_lunar_internal_model_path(dataset, seed, k)
    if not source_path.exists():
        raise FileNotFoundError(f"Expected LUNAR checkpoint not found: {source_path}")

    target_dir = MODELS_DIR / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"run_{run_index}_k_{k}_seed_{seed}.pth"
    shutil.copy2(source_path, target_path)
    print(f"Copied model checkpoint to: {target_path}")
    return target_path


def make_objective(train_x, train_y, val_x, val_y, dataset):
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
                dataset, SEED, k, samples, train_new_model=True,
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
    return objective


def fit_and_score_lunar(params, dataset, train_x, train_y, val_x, val_y, test_x, test_y):
    var.lr = params["lr"]
    var.wd = params["wd"]
    var.epsilon = params["epsilon"]
    var.proportion = params["proportion"]
    var.n_epochs = params["n_epochs"]

    original_device = var.device
    var.device = torch.device("cpu")

    try:
        start_train = __import__("time").time()
        out_val = LUNAR.run(
            train_x, train_y, val_x, val_y, val_x, val_y,
            dataset, SEED, params["k"], params["samples"], train_new_model=True,
        )
        runtime_train = __import__("time").time() - start_train
        scores_val = minmax_scale_scores(out_val.numpy())

        cleanup_memory()

        start_inference = __import__("time").time()
        out_test = LUNAR.run(
            train_x, train_y, val_x, val_y, test_x, test_y,
            dataset, SEED, params["k"], params["samples"], train_new_model=False,
        )
        runtime_inference = __import__("time").time() - start_inference
        scores_test = minmax_scale_scores(out_test.numpy())

        return scores_val, scores_test, runtime_train, runtime_inference
    finally:
        var.device = original_device
        cleanup_memory()


def run_optuna(dataset, run_cfg):
    run_index = run_cfg["run_index"]
    study_name = f"LUNAR_{dataset}_run{run_index}"

    train_x, train_y, val_x, val_y = make_optuna_subsample(
        dataset, SEED, run_cfg["n_train_opt"], run_cfg["n_val_opt"]
    )
    objective = make_objective(train_x, train_y, val_x, val_y, dataset)
    study = run_study(
        objective,
        study_name,
        SEED,
        N_TRIALS,
        results_dir=RESULTS_DIR,
        save_trials=True,
    )
    best_params = dict(study.best_params)

    del objective, study, train_x, train_y, val_x, val_y
    cleanup_memory()
    clear_dataset_cache()
    return best_params


def run_experiment(dataset, run_cfg, best_params):
    run_index = run_cfg["run_index"]
    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset,
        SEED,
        run_cfg["n_train_final"],
        run_cfg["n_val_final"],
        run_cfg["n_test_final"],
        max_nodes_budget=50_000_000,
        k=best_params["k"],
    )

    scores_val, scores_test, runtime_train, runtime_inference = fit_and_score_lunar(
        best_params, dataset, train_x, train_y, val_x, val_y, test_x, test_y
    )

    threshold_candidates = compare_threshold_strategies(val_y, scores_val, beta=2.0, normal_q=0.99)
    chosen_threshold_info = {"selection_method": "validation_f1_max", **threshold_candidates["f1_max"]}
    best_threshold = chosen_threshold_info["threshold"]

    print(f"[{dataset} run{run_index}] threshold candidates: {threshold_candidates}")
    print(f"[{dataset} run{run_index}] chosen threshold={best_threshold:.4f} by validation_f1_max")

    metrics = evaluate_scores(test_y, scores_test, threshold=best_threshold)
    print_metrics(f"LUNAR final - {dataset} run{run_index}", metrics)

    result_bundle = {
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
    copied_model_path = copy_lunar_model_to_models_dir(dataset, run_index, SEED, best_params["k"])

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
        model_path=copied_model_path,
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
