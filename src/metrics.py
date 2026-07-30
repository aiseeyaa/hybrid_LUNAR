import numpy as np
from sklearn.metrics import (
    precision_recall_curve, f1_score, fbeta_score, confusion_matrix,
    roc_auc_score, average_precision_score, precision_score, recall_score
)


def minmax_scale_scores(scores):
    scores = np.asarray(scores, dtype=float)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def _safe_prf(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(p), float(r), float(f1)


def find_best_f1_threshold(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1s = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        precisions[:-1] + recalls[:-1],
        out=np.zeros_like(thresholds, dtype=float),
        where=(precisions[:-1] + recalls[:-1]) > 0,
    )
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def find_best_fbeta_threshold(y_true, scores, beta=2.0):
    thresholds = np.unique(np.asarray(scores, dtype=float))
    if thresholds.size == 0:
        return 0.5, 0.0, 0.0, 0.0
    best = (0.5, -1.0, 0.0, 0.0)
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        p, r, _ = _safe_prf(y_true, y_pred)
        fb = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if fb > best[1]:
            best = (float(thr), float(fb), float(p), float(r))
    return best


def threshold_from_normal_quantile(y_true, scores, q=0.99):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    normal_scores = scores[y_true == 0]
    if normal_scores.size == 0:
        return 0.5
    return float(np.quantile(normal_scores, q))


def evaluate_scores(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    metrics = {
        "AUC_ROC": float(roc_auc_score(y_true, scores)),
        "AUC_PR": float(average_precision_score(y_true, scores)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }
    return metrics


def compare_threshold_strategies(val_y, scores_val, beta=2.0, normal_q=0.99):
    thr_f1, f1_val, p_f1, r_f1 = find_best_f1_threshold(val_y, scores_val)
    thr_fbeta, fbeta_val, p_fb, r_fb = find_best_fbeta_threshold(val_y, scores_val, beta=beta)
    thr_q = threshold_from_normal_quantile(val_y, scores_val, q=normal_q)
    y_pred_q = (scores_val >= thr_q).astype(int)
    p_q, r_q, f1_q = _safe_prf(val_y, y_pred_q)
    return {
        "f1_max": {
            "threshold": float(thr_f1), "score_name": "F1", "score_value": float(f1_val),
            "precision": float(p_f1), "recall": float(r_f1)
        },
        f"f{beta:g}_max": {
            "threshold": float(thr_fbeta), "score_name": f"F{beta:g}", "score_value": float(fbeta_val),
            "precision": float(p_fb), "recall": float(r_fb)
        },
        f"normal_q_{normal_q}": {
            "threshold": float(thr_q), "score_name": "F1", "score_value": float(f1_q),
            "precision": float(p_q), "recall": float(r_q)
        },
    }


def print_metrics(label, metrics):
    print(f"--- {label} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")