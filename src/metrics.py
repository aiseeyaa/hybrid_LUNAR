import numpy as np
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score, fbeta_score, confusion_matrix,
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


def find_min_recall_max_precision_threshold(y_true, scores, min_recall=0.90):
    """
    "Operational" threshold variant recommended in the review: fix a
    minimum acceptable recall (e.g. 0.90, i.e. catch at least 90% of
    attacks), then among all thresholds meeting that recall, pick the one
    that maximizes precision (fewest false alarms subject to the recall
    floor).

    If no threshold on the calibration set reaches min_recall (e.g. the
    model is too weak), we fall back to the threshold that achieves the
    highest recall actually attainable, and flag this in the returned dict
    via "recall_floor_met": False.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {
            "threshold": 0.5, "score_name": "Precision@MinRecall",
            "score_value": 0.0, "precision": 0.0, "recall": 0.0,
            "min_recall_target": float(min_recall), "recall_floor_met": False,
        }

    precisions_t = precisions[:-1]
    recalls_t = recalls[:-1]

    eligible = recalls_t >= min_recall
    if eligible.any():
        eligible_idx = np.where(eligible)[0]
        best_local = eligible_idx[np.argmax(precisions_t[eligible_idx])]
        recall_floor_met = True
    else:
        best_local = int(np.argmax(recalls_t))
        recall_floor_met = False

    return {
        "threshold": float(thresholds[best_local]),
        "score_name": "Precision@MinRecall",
        "score_value": float(precisions_t[best_local]),
        "precision": float(precisions_t[best_local]),
        "recall": float(recalls_t[best_local]),
        "min_recall_target": float(min_recall),
        "recall_floor_met": recall_floor_met,
    }


def recall_at_fpr(y_true, scores, target_fpr=0.01):
    """Threshold-agnostic-ish comparison metric: recall at a fixed FPR on
    the ROC curve (interpolated). Useful for comparing models on equal
    footing without committing to a single decision threshold."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def precision_at_recall(y_true, scores, target_recall=0.80):
    """Threshold-agnostic-ish comparison metric: best precision achievable
    while keeping recall at or above target_recall."""
    precisions, recalls, _ = precision_recall_curve(y_true, scores)
    eligible = recalls >= target_recall
    if not eligible.any():
        return 0.0
    return float(precisions[eligible].max())


def pr_curve_summary(y_true, scores, n_points=50):
    """
    Returns precision/recall/f1/threshold sampled at n_points evenly spaced
    thresholds, so the full precision-recall trade-off can be inspected and
    plotted later -- not just the single best-F1 point. Addresses the
    review's request to "record precision, recall and F1 for many
    thresholds, not just the best result".
    """
    scores = np.asarray(scores, dtype=float)
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    rows = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        p, r, f1 = _safe_prf(y_true, y_pred)
        rows.append({"threshold": float(thr), "precision": p, "recall": r, "f1": f1})
    return rows


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
        "Recall_at_FPR_0.01": recall_at_fpr(y_true, scores, target_fpr=0.01),
        "Precision_at_Recall_0.80": precision_at_recall(y_true, scores, target_recall=0.80),
    }
    return metrics


def compare_threshold_strategies(val_y, scores_val, beta=2.0, normal_q=0.99, min_recall=0.90):
    """
    Returns candidate thresholds under several selection rules, all computed
    on the (disjoint) calibration split, never on test data:
      - f1_max            : standard threshold, maximizes F1
      - f{beta}_max       : recall-oriented threshold, maximizes F-beta (beta>1 favors recall)
      - normal_q_{q}      : threshold at the q-th quantile of normal-class scores
      - operational       : minimum-recall-then-max-precision ("operational" variant from review)
    """
    thr_f1, f1_val, p_f1, r_f1 = find_best_f1_threshold(val_y, scores_val)
    thr_fbeta, fbeta_val, p_fb, r_fb = find_best_fbeta_threshold(val_y, scores_val, beta=beta)
    thr_q = threshold_from_normal_quantile(val_y, scores_val, q=normal_q)
    y_pred_q = (scores_val >= thr_q).astype(int)
    p_q, r_q, f1_q = _safe_prf(val_y, y_pred_q)
    operational = find_min_recall_max_precision_threshold(val_y, scores_val, min_recall=min_recall)

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
        "operational": operational,
    }


def print_metrics(label, metrics):
    print(f"--- {label} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
