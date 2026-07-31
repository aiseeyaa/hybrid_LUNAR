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
):
    return {
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
        "runtime_train": float(runtime_train),
        "runtime_inference": float(runtime_inference),
    }


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