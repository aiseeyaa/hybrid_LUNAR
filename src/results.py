import json
import uuid
from pathlib import Path

_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def ordinal_label(run_index: int) -> str:
    word = _ORDINALS.get(run_index, f"{run_index}th")
    return f"{word}_run"


def build_experiment_record(
    dataset_name,
    dataset_version,
    split_method,
    seed,
    preprocessing_version,
    model_type,
    fusion_strategy,
    hyperparameters,
    metrics,
    runtime_train,
    runtime_inference,
    threshold_info=None,
    model_path=None,
    threshold_variants=None,
    pr_curve=None,
):
    """
    threshold_variants: optional dict with the three reporting variants
    requested in the review -- e.g.
        {
            "standard": {...evaluate_scores() output at f1_max threshold...},
            "recall_oriented": {...evaluate_scores() output at f_beta threshold...},
            "operational": {...evaluate_scores() output at min-recall threshold...},
        }
    Each value is expected to be the dict returned by metrics.evaluate_scores.
    This lets every model_type report the same three operating points on the
    frozen test set under an identical protocol, instead of a single F1-max
    number.

    pr_curve: optional list of {threshold, precision, recall, f1} rows from
    metrics.pr_curve_summary, computed on the calibration split, so the full
    precision-recall trade-off can be inspected later rather than only the
    single best point.
    """
    record = {
        "experiment_id": str(uuid.uuid4()),
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "split_method": split_method,
        "seed": seed,
        "preprocessing_version": preprocessing_version,
        "model_type": model_type,
        "fusion_strategy": fusion_strategy,
        "hyperparameters": hyperparameters,
        "threshold": float(metrics["threshold"]),
        "threshold_info": threshold_info or {},
        "model_path": str(model_path) if model_path is not None else None,
        "AUC_ROC": float(metrics["AUC_ROC"]),
        "AUC_PR": float(metrics["AUC_PR"]),
        "Precision": float(metrics["Precision"]),
        "Recall": float(metrics["Recall"]),
        "F1": float(metrics["F1"]),
        "ConfusionMatrix": metrics["ConfusionMatrix"],
        "Recall_at_FPR_0.01": metrics.get("Recall_at_FPR_0.01"),
        "Precision_at_Recall_0.80": metrics.get("Precision_at_Recall_0.80"),
        "runtime_train": float(runtime_train),
        "runtime_inference": float(runtime_inference),
    }

    if threshold_variants is not None:
        record["threshold_variants"] = threshold_variants
    if pr_curve is not None:
        record["pr_curve"] = pr_curve

    return record


def save_record_json(record, results_dir, run_index, model_type, dataset_name):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    label = ordinal_label(run_index)
    filename = f"{label}_{model_type}_{dataset_name}.json"
    out_path = results_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Saved: {out_path}")
    return out_path
