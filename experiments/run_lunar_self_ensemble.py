"""
CO TU ROBIMY (Eksperyment 3 z drabiny ablacyjnej, BEZ WLASNEGO TUNINGU):
Homogeniczny ensemble - kilka niezaleznych instancji TEGO SAMEGO LUNAR-a
(bootstrapowe proby danych treningowych) laczonych 4 strategiami
zespolowymi: bagging, boosting, voting, stacking.

ZMIANA: bazowe hiperparametry LUNAR-a (k, samples, lr, wd, epsilon,
proportion, n_epochs) wczytujemy z run_single_experiment.py zamiast
tunowac je od nowa. Wszystkie instancje self-ensemble uzywaja TYCH SAMYCH
hiperparametrow bazowych, roznia sie tylko probka treningowa i ziarnem
losowym - dokladnie tak jak w prawdziwym baggingu/boostingu (tam tez
architektura/hiperparametry sa ustalone, zmienia sie tylko dane).

Logika budowania 4 strategii jest we wspolnym module
self_ensemble_builders.py (reuzywany tez przez
run_robustness_self_ensemble_probe.py).

WYMAGA wczesniej uruchomionego: run_single_experiment.py dla tego samego
(dataset, run_index).

Nie modyfikuje LUNAR.py, utils.py ani variables.py.
"""

import sys
import gc
import argparse
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
EXTERNAL_LUNAR_DIR = ROOT / "external" / "LUNAR"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

sys.path.append(str(SRC_DIR))
sys.path.append(str(EXTERNAL_LUNAR_DIR))

from data import make_final_subsample, clear_dataset_cache
from metrics import print_metrics
from results import build_experiment_record, save_record_json
from threshold_reporting import calibrate_and_evaluate
from lunar_params_loading import load_lunar_params
from self_ensemble_builders import build_self_ensemble, cleanup_memory

SEED = 29
N_INSTANCES = 5
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "LUNAR_Self_Ensemble"

MIN_RECALL_TARGET = 0.90
FBETA = 2.0
NORMAL_Q = 0.99

STRATEGIES = ["bagging", "boosting", "voting", "stacking"]

RUN_CONFIGS = {
    1: dict(run_index=1, n_train_final=154000, n_val_final=66000, n_test_final=100000, notes="run1"),
    2: dict(run_index=2, n_train_final=154000, n_val_final=66000, n_test_final=100000, notes="run2"),
    3: dict(run_index=3, n_train_final=154000, n_val_final=66000, n_test_final=100000, notes="run3"),
}


def run_experiment(dataset, run_cfg, lunar_params):
    run_index = run_cfg["run_index"]
    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=lunar_params["k"],
    )

    variant_records = {}
    for strategy in STRATEGIES:
        v_scores, t_scores, tr, inf, info = build_self_ensemble(
            strategy, lunar_params, dataset, SEED, N_INSTANCES,
            train_x, train_y, val_x, val_y, test_x, test_y
        )
        metrics, threshold_info, threshold_variants, pr_curve, threshold_candidates = calibrate_and_evaluate(
            val_y, v_scores, test_y, t_scores,
            beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
        )
        print_metrics(f"{MODEL_TYPE} [{strategy}] - {dataset} run{run_index}", metrics)
        variant_records[strategy] = {
            "metrics": metrics, "threshold_info": threshold_info, "threshold_variants": threshold_variants,
            "pr_curve": pr_curve, "runtime_train": tr, "runtime_inference": inf, "extra": info,
        }
        cleanup_memory()

    del train_x, train_y, val_x, val_y, test_x, test_y
    cleanup_memory()
    clear_dataset_cache()
    return variant_records


def save_results(dataset, run_cfg, lunar_params, variant_records):
    run_index = run_cfg["run_index"]
    saved = []
    for strategy_name, variant in variant_records.items():
        record = build_experiment_record(
            dataset_name=dataset,
            dataset_version=DATASET_VERSION,
            split_method=SPLIT_METHOD,
            seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION,
            model_type=MODEL_TYPE,
            fusion_strategy=strategy_name,
            hyperparameters={**lunar_params, "n_instances": N_INSTANCES, "extra": variant["extra"]},
            metrics=variant["metrics"],
            runtime_train=variant["runtime_train"],
            runtime_inference=variant["runtime_inference"],
            threshold_info=variant["threshold_info"],
            threshold_variants=variant["threshold_variants"],
            pr_curve=variant["pr_curve"],
        )
        tag = f"{MODEL_TYPE}_{strategy_name}"
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
    variant_records = run_experiment(args.dataset, run_cfg, lunar_params)
    saved = save_results(args.dataset, run_cfg, lunar_params, variant_records)

    best_f1 = max(r["F1"] for r in saved)
    print(f"Finished {args.dataset} run {args.run_index}: {len(saved)} strategie self-ensemble, best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
