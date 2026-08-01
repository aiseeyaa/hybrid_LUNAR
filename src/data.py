import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append("../external/LUNAR")
import utils


@lru_cache(maxsize=1)
def load_dataset_cached(dataset: str, seed: int):
    return utils.load_dataset(dataset, seed)


@lru_cache(maxsize=1)
def load_split_with_calibration_cached(dataset: str, seed: int, calib_fraction: float = 0.5):
    """
    Splits the original validation set into two DISJOINT parts, once per
    (dataset, seed):
      - val_tune   -> used only by Optuna for hyperparameter search
      - val_calib  -> used only for threshold calibration (never for tuning)

    This avoids reusing (parts of) the same validation rows for both
    hyperparameter selection and threshold selection, which was the main
    methodological concern raised in the review. The test set is left
    completely untouched here and is only ever subsampled (never split)
    downstream, so it stays frozen for final evaluation.

    The natural class ratio of the original validation set is preserved in
    both val_tune and val_calib (stratified split, no oversampling).
    """
    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset_cached(dataset, seed)

    stratify = val_y if len(np.unique(val_y)) > 1 else None
    val_tune_x, val_calib_x, val_tune_y, val_calib_y = train_test_split(
        val_x, val_y, test_size=calib_fraction, stratify=stratify, random_state=seed
    )

    return train_x, train_y, val_tune_x, val_tune_y, val_calib_x, val_calib_y, test_x, test_y


def clear_dataset_cache():
    load_dataset_cached.cache_clear()
    load_split_with_calibration_cached.cache_clear()


def _stratified_subsample(x, y, n_target, seed, min_minority):
    if len(x) <= n_target:
        return x, y
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) > 1 and counts.min() >= 2:
        frac_minority = counts.min() / len(y)
        n_needed = max(n_target, int(min_minority / frac_minority))
        n_eff = min(n_needed, len(x))
        x_sub, _, y_sub, _ = train_test_split(
            x, y, train_size=n_eff, stratify=y, random_state=seed
        )
    else:
        x_sub, y_sub = x, y
    return x_sub, y_sub


def make_optuna_subsample(dataset, seed, n_train, n_val, calib_fraction=0.5, min_minority=50):
    """
    Subsample used ONLY for Optuna hyperparameter tuning. Draws from
    val_tune (the tuning half of the validation split), never from
    val_calib, so the calibration data used later for threshold selection
    is never seen during hyperparameter search.
    """
    train_x, train_y, val_tune_x, val_tune_y, _, _, _, _ = load_split_with_calibration_cached(
        dataset, seed, calib_fraction
    )

    if len(train_x) > n_train:
        stratify = train_y if len(np.unique(train_y)) > 1 else None
        train_x_sub, _, train_y_sub, _ = train_test_split(
            train_x, train_y, train_size=n_train,
            stratify=stratify, random_state=seed
        )
    else:
        train_x_sub, train_y_sub = train_x, train_y

    if len(val_tune_x) > n_val:
        classes, counts = np.unique(val_tune_y, return_counts=True)
        min_class_count = counts.min() if len(classes) > 1 else 0

        if len(classes) > 1 and min_class_count >= 2:
            frac_minority = min_class_count / len(val_tune_y)
            required_n_val = max(n_val, int(min_minority / frac_minority))
            n_val_eff = min(required_n_val, len(val_tune_x))
            val_x_sub, _, val_y_sub, _ = train_test_split(
                val_tune_x, val_tune_y, train_size=n_val_eff,
                stratify=val_tune_y, random_state=seed
            )
        else:
            val_x_sub, val_y_sub = val_tune_x, val_tune_y
    else:
        val_x_sub, val_y_sub = val_tune_x, val_tune_y

    n_pos = int(val_y_sub.sum())
    print(f"{dataset} optuna(tune) subsample -> train: {len(train_x_sub)}, "
          f"val_tune: {len(val_x_sub)} (val_tune positives: {n_pos})")

    return train_x_sub, train_y_sub, val_x_sub, val_y_sub


def make_final_subsample(dataset, seed, n_train, n_val, n_test,
                          calib_fraction=0.5, min_minority=40,
                          max_nodes_budget=None, k=1):
    """
    Subsample used for the final model fit + threshold calibration + frozen
    test evaluation. The "val" returned here is val_calib (the calibration
    half of the validation split) -- disjoint from the val_tune data used
    by make_optuna_subsample. Natural class ratio is preserved (no
    oversampling) in line with the review recommendation.
    """
    train_x, train_y, _, _, val_calib_x, val_calib_y, test_x, test_y = load_split_with_calibration_cached(
        dataset, seed, calib_fraction
    )

    if max_nodes_budget is not None:
        max_safe_nodes = int(max_nodes_budget / max(k, 1))
        n_train = min(n_train, max_safe_nodes)
        n_val = min(n_val, int(max_safe_nodes * 0.2))
        n_test = min(n_test, int(max_safe_nodes * 0.2))

    if len(train_x) > n_train:
        stratify = train_y if len(np.unique(train_y)) > 1 else None
        train_x, _, train_y, _ = train_test_split(
            train_x, train_y, train_size=n_train,
            stratify=stratify, random_state=seed
        )

    val_x, val_y = _stratified_subsample(val_calib_x, val_calib_y, n_val, seed, min_minority)
    test_x, test_y = _stratified_subsample(test_x, test_y, n_test, seed, min_minority)

    print(f"{dataset} final subsample -> train: {len(train_x)}, val_calib: {len(val_x)}, "
          f"test: {len(test_x)} (test positives: {int(test_y.sum())})")

    return train_x, train_y, val_x, val_y, test_x, test_y
