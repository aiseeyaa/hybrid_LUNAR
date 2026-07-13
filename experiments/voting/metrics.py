import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def minmax_scale_scores(scores):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    smin = scores.min()
    smax = scores.max()
    if smax - smin < 1e-12:
        return np.zeros_like(scores)
    return (scores - smin) / (smax - smin)

def find_best_threshold(y_true, y_score, num_thresholds=200):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).reshape(-1)

    thresholds = np.linspace(y_score.min(), y_score.max(), num_thresholds)
    best_thr = thresholds[0]
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_thr), float(best_f1)

def evaluate_scores(y_true, y_score, threshold=None):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).reshape(-1)

    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def print_metrics(title, metrics_dict):
    print(f"\n===== {title} =====")
    print(f"AUC:       {metrics_dict['auc']:.4f}")
    print(f"PR-AUC:    {metrics_dict['pr_auc']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall:    {metrics_dict['recall']:.4f}")
    print(f"F1-score:  {metrics_dict['f1']:.4f}")
    print(f"Threshold: {metrics_dict['threshold']:.6f}")
    print(f"TN: {metrics_dict['tn']} | FP: {metrics_dict['fp']} | FN: {metrics_dict['fn']} | TP: {metrics_dict['tp']}")