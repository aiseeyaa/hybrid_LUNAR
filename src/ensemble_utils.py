
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from optuna_utils import run_study
from metrics import minmax_scale_scores


def fit_dbscan_ref(train_x, eps, min_samples, metric):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    db.fit(train_x)
    if len(db.core_sample_indices_) == 0:
        return None
    core = np.asarray(train_x)[db.core_sample_indices_]
    nn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1)
    nn.fit(core)
    return nn


def tune_if(train_x, val_x, val_y, seed, n_trials, results_dir):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "contamination": trial.suggest_float("contamination", 0.01, 0.5),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        clf = IsolationForest(**params, random_state=seed, n_jobs=-1)
        clf.fit(train_x)
        return roc_auc_score(val_y, -clf.score_samples(val_x))
    return run_study(objective, "IF_tmp", seed, n_trials, results_dir=results_dir).best_params


def tune_lof(train_x, val_x, val_y, seed, n_trials, results_dir):
    def objective(trial):
        metric = trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"])
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 100, log=True),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60, step=10),
            "metric": metric,
            "contamination": trial.suggest_float("contamination", 0.01, 0.5),
            "p": trial.suggest_int("p", 1, 2) if metric == "minkowski" else 2,
            "novelty": True,
            "n_jobs": -1,
        }
        clf = LocalOutlierFactor(**params)
        clf.fit(train_x)
        return roc_auc_score(val_y, -clf.score_samples(val_x))
    return run_study(objective, "LOF_tmp", seed, n_trials, results_dir=results_dir).best_params


def tune_ocsvm(train_x, val_x, val_y, seed, n_trials, results_dir):
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
        params = {
            "kernel": kernel,
            "nu": trial.suggest_float("nu", 0.01, 0.5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "degree": trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3,
            "coef0": trial.suggest_float("coef0", 0.0, 1.0) if kernel in ("poly", "sigmoid") else 0.0,
        }
        clf = OneClassSVM(**params)
        clf.fit(train_x)
        return roc_auc_score(val_y, -clf.decision_function(val_x))
    return run_study(objective, "OCSVM_tmp", seed, n_trials, results_dir=results_dir).best_params


def tune_dbscan(train_x, val_x, val_y, seed, n_trials, results_dir):
    def objective(trial):
        params = {
            "eps": trial.suggest_float("eps", 0.05, 5.0, log=True),
            "min_samples": trial.suggest_int("min_samples", 3, 50, log=True),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        }
        nn = fit_dbscan_ref(train_x, **params)
        if nn is None:
            raise optuna.exceptions.TrialPruned()
        return roc_auc_score(val_y, nn.kneighbors(val_x, n_neighbors=1)[0].ravel())
    return run_study(objective, "DBSCAN_tmp", seed, n_trials, results_dir=results_dir).best_params


def score_if(params, train_x, val_x, test_x, seed):
    clf = IsolationForest(**params, random_state=seed, n_jobs=-1)
    t0 = time.time(); clf.fit(train_x); tr = time.time() - t0
    t1 = time.time(); val = -clf.score_samples(val_x); test = -clf.score_samples(test_x); inf = time.time() - t1
    return minmax_scale_scores(val), minmax_scale_scores(test), tr, inf


def score_lof(params, train_x, val_x, test_x):
    clf = LocalOutlierFactor(**params, novelty=True, n_jobs=-1)
    t0 = time.time(); clf.fit(train_x); tr = time.time() - t0
    t1 = time.time(); val = -clf.score_samples(val_x); test = -clf.score_samples(test_x); inf = time.time() - t1
    return minmax_scale_scores(val), minmax_scale_scores(test), tr, inf


def score_ocsvm(params, train_x, val_x, test_x):
    clf = OneClassSVM(**params)
    t0 = time.time(); clf.fit(train_x); tr = time.time() - t0
    t1 = time.time(); val = -clf.decision_function(val_x); test = -clf.decision_function(test_x); inf = time.time() - t1
    return minmax_scale_scores(val), minmax_scale_scores(test), tr, inf


def score_dbscan(params, train_x, val_x, test_x):
    t0 = time.time(); nn = fit_dbscan_ref(train_x, **params); tr = time.time() - t0
    if nn is None:
        raise RuntimeError("DBSCAN produced zero core samples")
    t1 = time.time(); val = nn.kneighbors(val_x, n_neighbors=1)[0].ravel(); test = nn.kneighbors(test_x, n_neighbors=1)[0].ravel(); inf = time.time() - t1
    return minmax_scale_scores(val), minmax_scale_scores(test), tr, inf


def rank_average(arr2d):
    ranks = np.vstack([pd.Series(col).rank(method="average").to_numpy() for col in arr2d.T]).T
    return minmax_scale_scores(ranks.mean(axis=1))


def tune_meta_fusion(val_matrix, val_y, seed, n_trials, results_dir, fusion_strategies):
    def objective(trial):
        strategy = trial.suggest_categorical("fusion_strategy", fusion_strategies)
        if strategy == "mean":
            fused = val_matrix.mean(axis=1)
        elif strategy == "max":
            fused = val_matrix.max(axis=1)
        elif strategy == "rank_mean":
            fused = rank_average(val_matrix)
        elif strategy == "weighted":
            weights_raw = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(val_matrix.shape[1])])
            weights = weights_raw / max(weights_raw.sum(), 1e-12)
            fused = val_matrix @ weights
        else:
            c = trial.suggest_float("meta_C", 1e-2, 1e2, log=True)
            meta = LogisticRegression(C=c, max_iter=3000, random_state=seed)
            meta.fit(val_matrix, val_y)
            fused = meta.predict_proba(val_matrix)[:, 1]
        return roc_auc_score(val_y, fused)
    study = run_study(objective, "META_tmp", seed, n_trials, results_dir=results_dir)
    return study.best_params


def apply_meta_fusion(best_meta, val_matrix, test_matrix, val_y, seed):
    strategy = best_meta["fusion_strategy"]
    if strategy == "mean":
        return val_matrix.mean(axis=1), test_matrix.mean(axis=1)
    if strategy == "max":
        return val_matrix.max(axis=1), test_matrix.max(axis=1)
    if strategy == "rank_mean":
        return rank_average(val_matrix), rank_average(test_matrix)
    if strategy == "weighted":
        w_keys = sorted([k for k in best_meta if k.startswith("w_")], key=lambda x: int(x.split("_")[1]))
        weights_raw = np.array([best_meta[k] for k in w_keys])
        weights = weights_raw / max(weights_raw.sum(), 1e-12)
        return val_matrix @ weights, test_matrix @ weights
    meta = LogisticRegression(C=best_meta["meta_C"], max_iter=3000, random_state=seed)
    meta.fit(val_matrix, val_y)
    return meta.predict_proba(val_matrix)[:, 1], meta.predict_proba(test_matrix)[:, 1]
