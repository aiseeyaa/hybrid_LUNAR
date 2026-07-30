import optuna
from optuna.samplers import TPESampler
import pandas as pd
from pathlib import Path


def run_study(objective_fn, study_name, seed, n_trials, catch=(RuntimeError,),
              results_dir=None, save_trials=True):
    """Generic Optuna search wrapper shared by every model notebook."""
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True, catch=catch)

    print(f"Best AUC ({study_name}): {study.best_value:.4f}")
    print(f"Best params ({study_name}): {study.best_params}")

    if save_trials and results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(results_dir / f"optuna_trials_{study_name}.csv", index=False)

    return study