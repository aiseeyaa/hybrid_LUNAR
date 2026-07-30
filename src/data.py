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

def clear_dataset_cache():
    load_dataset_cached.cache_clear()


def make_optuna_subsample(dataset, seed, n_train, n_val, min_minority=50):
    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset_cached(dataset, seed)

    if len(train_x) > n_train:
        stratify = train_y if len(np.unique(train_y)) > 1 else None
        train_x_sub, _, train_y_sub, _ = train_test_split(
            train_x, train_y, train_size=n_train,
            stratify=stratify, random_state=seed
        )
    else:
        train_x_sub, train_y_sub = train_x, train_y

    if len(val_x) > n_val:
        classes, counts = np.unique(val_y, return_counts=True)
        min_class_count = counts.min() if len(classes) > 1 else 0

        if len(classes) > 1 and min_class_count >= 2:
            frac_minority = min_class_count / len(val_y)
            required_n_val = max(n_val, int(min_minority / frac_minority))
            n_val_eff = min(required_n_val, len(val_x))
            val_x_sub, _, val_y_sub, _ = train_test_split(
                val_x, val_y, train_size=n_val_eff,
                stratify=val_y, random_state=seed
            )
        else:
            val_x_sub, val_y_sub = val_x, val_y
    else:
        val_x_sub, val_y_sub = val_x, val_y

    n_pos = int(val_y_sub.sum())
    print(f"{dataset} optuna subsample -> train: {len(train_x_sub)}, "
          f"val: {len(val_x_sub)} (val positives: {n_pos})")

    return train_x_sub, train_y_sub, val_x_sub, val_y_sub


def make_final_subsample(dataset, seed, n_train, n_val, n_test,
                          min_minority=40, max_nodes_budget=None, k=1):
    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset_cached(dataset, seed)

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

    def stratified_subsample(x, y, n_target):
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

    val_x, val_y = stratified_subsample(val_x, val_y, n_val)
    test_x, test_y = stratified_subsample(test_x, test_y, n_test)

    print(f"{dataset} final subsample -> train: {len(train_x)}, val: {len(val_x)}, "
          f"test: {len(test_x)} (test positives: {int(test_y.sum())})")

    return train_x, train_y, val_x, val_y, test_x, test_y