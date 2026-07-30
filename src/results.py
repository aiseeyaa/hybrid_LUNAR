import json
import uuid
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_score, recall_score

_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd"}


def ordinal_label(run_index: int) -> str:
    word = _ORDINALS.get(run_index, f"{run_index}th")
    return f"{word}_run"


def build_experiment_record(
    dataset_name, dataset_version, split_method, seed, preprocessing_version,
    model_type, fusion_strategy, hyperparameters,
    threshold, scores_test, test_y,
    runtime_train, runtime_inference, memory_peak,
    notes="", threshold_info=None, model_path=None,
):
    y_true = np.asarray(test_y)
    scores_test = np.asarray(scores_test)
    y_pred = (scores_test >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, scores_test)
    auc_pr = average_precision_score(y_true, scores_test)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

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
        "threshold": float(threshold),
        "threshold_info": threshold_info or {},
        "model_path": str(model_path) if model_path is not None else None,
        "AUC_ROC": float(auc_roc),
        "AUC_PR": float(auc_pr),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "ConfusionMatrix": cm.tolist(),
        "runtime_train": float(runtime_train),
        "runtime_inference": float(runtime_inference),
        "memory_peak": float(memory_peak),
        "notes": notes,
    }


def save_record_json(record, results_dir, run_index, model_type, dataset_name):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    label = ordinal_label(run_index)
    filename = f"{label}_{model_type}_{dataset_name}.json"
    out_path = results_dir / filename
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved: {out_path}")
    return out_path