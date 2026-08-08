"""
CO TU ROBIMY:
Analogiczny modul do baseline_selection.py, ale dla eksperymentu 3
(run_lunar_self_ensemble.py). Czyta juz zapisane wyniki 4 strategii
self-ensemble LUNAR-a (bagging, boosting, voting, stacking) i wybiera
empirycznie te o najwyzszym F1 dla danego (dataset, run_index).

Uzywane przez run_fusion_level_comparison.py, zeby zestawic "najlepszy
self-ensemble LUNAR-a" obok "LUNAR + najlepszy klasyczny baseline" w jednym
porownaniu sposobow laczenia modeli - nie tylko poziomow fuzji dla jednej
pary modeli.

Wymaga, zeby run_lunar_self_ensemble.py zostal juz uruchomiony dla danego
(dataset, run_index) - w przeciwnym razie rzuca czytelny blad.
"""

import json
from pathlib import Path

from results import ordinal_label

SELF_ENSEMBLE_MODEL_TYPE = "LUNAR_Self_Ensemble"
SELF_ENSEMBLE_STRATEGIES = ["bagging", "boosting", "voting", "stacking"]


def load_self_ensemble_record(results_dir, run_index, strategy, dataset):
    tag = f"{SELF_ENSEMBLE_MODEL_TYPE}_{strategy}"
    path = Path(results_dir) / f"{ordinal_label(run_index)}_{tag}_{dataset}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_best_self_ensemble(results_dir, run_index, dataset, metric="F1"):
    """
    Zwraca (strategy_name, record) dla strategii self-ensemble LUNAR-a
    (bagging/boosting/voting/stacking) o najwyzszym `metric` (domyslnie F1
    na zamrozonym zbiorze testowym) dla danego (dataset, run_index).
    """
    found = {}
    for strategy in SELF_ENSEMBLE_STRATEGIES:
        rec = load_self_ensemble_record(results_dir, run_index, strategy, dataset)
        if rec is not None:
            found[strategy] = rec

    if not found:
        raise FileNotFoundError(
            f"Brak wynikow self-ensemble LUNAR-a dla dataset={dataset}, run_index={run_index}. "
            f"Najpierw uruchom run_lunar_self_ensemble.py dla tego datasetu i run_index."
        )

    best_strategy = max(found, key=lambda s: found[s][metric])
    print(f"[self_ensemble_selection] najlepsza strategia self-ensemble LUNAR dla {dataset} run{run_index}: "
          f"{best_strategy} ({metric}={found[best_strategy][metric]:.4f})")
    return best_strategy, found[best_strategy]
