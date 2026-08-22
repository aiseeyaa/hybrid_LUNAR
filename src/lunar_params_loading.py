# modul do wczytywania JUZ DOSTROJONYCH hiperparametrow LUNAR-a z wyniku run_single_experiment.py 

import json
from pathlib import Path

from results import ordinal_label

LUNAR_MODEL_TYPE = "LUNAR"


def load_lunar_params(results_dir, run_index, dataset):
    """
    Zwraca slownik hiperparametrow LUNAR-a (k, samples, lr, wd, epsilon,
    proportion, n_epochs) zapisany przez run_single_experiment.py dla
    danego (dataset, run_index).
    """
    path = Path(results_dir) / f"{ordinal_label(run_index)}_{LUNAR_MODEL_TYPE}_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Brak wynikow solo LUNAR dla dataset={dataset}, run_index={run_index}. "
            f"Najpierw uruchom run_single_experiment.py dla tego datasetu i run_index."
        )
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    print(f"[lunar_params_loading] wczytano hiperparametry LUNAR z run_single_experiment.py "
          f"dla {dataset} run{run_index}: {record['hyperparameters']}")
    return record["hyperparameters"]
