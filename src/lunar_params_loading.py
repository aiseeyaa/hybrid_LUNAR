"""
CO TU ROBIMY:
Wspolny modul do wczytywania JUZ DOSTROJONYCH hiperparametrow LUNAR-a z
wyniku run_single_experiment.py (eksperyment solo LUNAR), zamiast ponownego
tunowania Optuna w kazdym skrypcie, ktory LUNAR-a tylko UZYWA (ensemble,
self-ensemble, fusion-level, robustness itd.).

Powod: strojenie (Optuna, 200 prob) jest kosztowne i powinno sie odbyc
DOKLADNIE RAZ na (dataset, run_index) - w run_single_experiment.py. Wszystkie
pozostale skrypty, ktore LACZA LUNAR-a z czyms innym albo analizuja jego
zachowanie (self-ensemble, fuzja, odpornosc), powinny reuzywac te same,
juz znalezione hiperparametry - nie szukac ich na nowo za kazdym razem.

Wymaga, zeby run_single_experiment.py zostal juz uruchomiony dla danego
(dataset, run_index) - w przeciwnym razie rzuca czytelny blad z instrukcja
co odpalic.
"""

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
