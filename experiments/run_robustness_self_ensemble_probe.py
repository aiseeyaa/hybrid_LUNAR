"""
CO TU ROBIMY (BEZ WLASNEGO TUNINGU):
Sparametryzowany, POJEDYNCZY punkt analizy odpornosci dla self-ensemble
LUNAR-a (dowolna z 4 strategii, dowolny czynnik/wartosc zaklocenia).

ZMIANA: hiperparametry LUNAR-a wczytujemy z run_single_experiment.py, a
(opcjonalnie, przy --compare_baseline) hiperparametry klasycznego
baseline'u z jego solo-wyniku. Jedyne tunowanie w tym skrypcie to (przy
--compare_baseline) dobor strategii fuzji dla pary LUNAR+baseline.

Przyklady uzycia:
  python run_robustness_self_ensemble_probe.py CICIDS 1 --factor noise --value 0.15 --strategy boosting
  python run_robustness_self_ensemble_probe.py CICIDS 1 --factor feature_dropout --value 0.30 --strategy stacking --n_instances 7 --compare_baseline

WYMAGA wczesniej uruchomionego: run_single_experiment.py (zawsze), oraz
run_isolation_forest.py / run_lof.py / run_ocsvm.py (tylko
jesli podano --compare_baseline).

"""

import sys
import argparse
from pathlib import Path

import numpy as np
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
from ensemble_utils import tune_meta_fusion, apply_meta_fusion
from baseline_selection import select_best_baseline, BASELINE_REGISTRY
from lunar_params_loading import load_lunar_params
from self_ensemble_builders import build_self_ensemble, fit_lunar_instance, cleanup_memory

SEED = 29
META_TRIALS = 80
FUSION_STRATEGIES = ["mean", "max", "weighted", "rank_mean", "stacking_lr"]
DATASET_VERSION = "v1"
PREPROCESSING_VERSION = "v1"
SPLIT_METHOD = "stratified_train_val_calib_test_fixed_seed"
MODEL_TYPE = "Robustness_SelfEnsemble_Probe"

MIN_RECALL_TARGET = 0.90
FBETA = 2.0
NORMAL_Q = 0.99

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


def apply_factor(factor, value, train_x, train_y, val_x, val_y, test_x, test_y, seed):
    if factor == "none":
        return train_x, train_y, val_x, val_y, test_x, test_y
    if factor == "noise":
        return add_noise(train_x, value, seed), train_y, val_x, val_y, test_x, test_y
    if factor == "imbalance_factor_test":
        test_x_mod, test_y_mod = downsample_anomalies(test_x, test_y, value, seed)
        return train_x, train_y, val_x, val_y, test_x_mod, test_y_mod
    if factor == "feature_dropout":
        return (feature_dropout(train_x, value, seed), train_y,
                feature_dropout(val_x, value, seed), val_y,
                feature_dropout(test_x, value, seed), test_y)
    raise ValueError(f"Nieznany factor: {factor}")


def run_experiment(args, run_cfg, lunar_params):
    dataset, run_index = args.dataset, run_cfg["run_index"]

    train_x, train_y, val_x, val_y, test_x, test_y = make_final_subsample(
        dataset, SEED, run_cfg["n_train_final"], run_cfg["n_val_final"], run_cfg["n_test_final"],
        max_nodes_budget=50_000_000, k=lunar_params["k"],
    )

    x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p = apply_factor(
        args.factor, args.value, train_x, train_y, val_x, val_y, test_x, test_y, SEED
    )

    results = {}

    se_val, se_test, se_tr, se_inf, se_info = build_self_ensemble(
        args.strategy, lunar_params, dataset, SEED, args.n_instances,
        x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p
    )
    se_metrics, se_thr_info, se_thr_variants, se_pr_curve, _ = calibrate_and_evaluate(
        y_val_p, se_val, y_test_p, se_test, beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
    )
    print_metrics(
        f"{MODEL_TYPE} [self_ensemble={args.strategy}, {args.factor}={args.value}] - {dataset} run{run_index}",
        se_metrics,
    )
    results["lunar_self_ensemble_" + args.strategy] = {
        "metrics": se_metrics, "threshold_info": se_thr_info, "threshold_variants": se_thr_variants,
        "pr_curve": se_pr_curve, "runtime_train": se_tr, "runtime_inference": se_inf,
        "extra": se_info, "combination_type": f"lunar_self_ensemble_{args.strategy}",
    }

    if args.compare_baseline:
        best_model_type, best_record = select_best_baseline(RESULTS_DIR, run_index, dataset)
        best_baseline_params = best_record["hyperparameters"]
        baseline_scorer = BASELINE_REGISTRY[best_model_type]["scorer"]

        lunar_val, lunar_test, tr_lunar, inf_lunar = fit_lunar_instance(
            lunar_params, dataset, SEED, x_train_p, y_train_p, x_val_p, y_val_p, x_test_p, y_test_p
        )
        base_val, base_test, tr_base, inf_base = baseline_scorer(
            best_baseline_params, x_train_p, x_val_p, x_test_p, SEED
        )
        val_matrix = np.column_stack([lunar_val, base_val])
        test_matrix = np.column_stack([lunar_test, base_test])
        best_meta = tune_meta_fusion(val_matrix, y_val_p, SEED, META_TRIALS, RESULTS_DIR, FUSION_STRATEGIES)
        fused_val, fused_test = apply_meta_fusion(best_meta, val_matrix, test_matrix, y_val_p, SEED)

        pair_metrics, pair_thr_info, pair_thr_variants, pair_pr_curve, _ = calibrate_and_evaluate(
            y_val_p, fused_val, y_test_p, fused_test, beta=FBETA, normal_q=NORMAL_Q, min_recall=MIN_RECALL_TARGET,
        )
        print_metrics(
            f"{MODEL_TYPE} [lunar_plus_{best_model_type}, {args.factor}={args.value}] - {dataset} run{run_index}",
            pair_metrics,
        )
        results[f"lunar_plus_{best_model_type.lower()}"] = {
            "metrics": pair_metrics, "threshold_info": pair_thr_info, "threshold_variants": pair_thr_variants,
            "pr_curve": pair_pr_curve, "runtime_train": tr_lunar + tr_base, "runtime_inference": inf_lunar + inf_base,
            "extra": {"fusion_strategy": best_meta["fusion_strategy"], "meta": best_meta,
                      "best_baseline_model_type": best_model_type, "best_baseline_params": best_baseline_params},
            "combination_type": f"lunar_plus_{best_model_type.lower()}",
        }

    del train_x, train_y, val_x, val_y, test_x, test_y
    cleanup_memory()
    clear_dataset_cache()
    return results


def save_results(dataset, run_cfg, lunar_params, args, results):
    run_index = run_cfg["run_index"]
    saved = []
    for combo_name, payload in results.items():
        record = build_experiment_record(
            dataset_name=dataset, dataset_version=DATASET_VERSION, split_method=SPLIT_METHOD, seed=SEED,
            preprocessing_version=PREPROCESSING_VERSION, model_type=MODEL_TYPE, fusion_strategy=combo_name,
            hyperparameters={
                "lunar": lunar_params, "factor": args.factor, "value": args.value, "n_instances": args.n_instances,
                "combination_type": payload["combination_type"], "extra": payload["extra"],
            },
            metrics=payload["metrics"], runtime_train=payload["runtime_train"], runtime_inference=payload["runtime_inference"],
            threshold_info=payload["threshold_info"], threshold_variants=payload["threshold_variants"], pr_curve=payload["pr_curve"],
        )
        tag = f"{MODEL_TYPE}_{combo_name}_{args.factor}_{args.value}"
        save_record_json(record, RESULTS_DIR, run_index, tag, dataset)
        saved.append(record)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Pojedynczy, sparametryzowany punkt analizy odpornosci self-ensemble LUNAR-a.")
    parser.add_argument("dataset", choices=["CICIDS", "UNSW_NB15"])
    parser.add_argument("run_index", type=int, choices=sorted(RUN_CONFIGS))
    parser.add_argument("--factor", choices=["none", "noise", "imbalance_factor_test", "feature_dropout"], default="none")
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--strategy", choices=["bagging", "boosting", "voting", "stacking"], required=True)
    parser.add_argument("--n_instances", type=int, default=5)
    parser.add_argument("--compare_baseline", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    run_cfg = RUN_CONFIGS[args.run_index]
    lunar_params = load_lunar_params(RESULTS_DIR, args.run_index, args.dataset)
    results = run_experiment(args, run_cfg, lunar_params)
    saved = save_results(args.dataset, run_cfg, lunar_params, args, results)

    print(f"\nFinished {args.dataset} run {args.run_index} "
          f"[factor={args.factor}={args.value}, strategy={args.strategy}, n_instances={args.n_instances}]:")
    for r in saved:
        print(f"  {r['fusion_strategy']}: F1={r['F1']:.4f}, AUC_ROC={r['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
