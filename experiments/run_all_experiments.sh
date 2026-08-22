#!/bin/bash
# CO TU ROBIMY:
# Uruchamia WSZYSTKIE eksperymenty (oprocz run_robustness_self_ensemble_probe.py,
# ktory jest reczny/parametryzowany) dla obu datasetow i wszystkich 3 run_index,
# w kolejnosci wymaganej przez zaleznosci miedzy skryptami:
#   Faza 1: modele solo (LUNAR + 3 klasyczne) - jedyne miejsca ze strojeniem Optuna
#   Faza 2: ensemble/analizy zalezne TYLKO od Fazy 1
#   Faza 3: eksperymenty zalezne od Fazy 2 (self-ensemble)
#   Faza 4: agregacja wszystkich wynikow
#
# Z katalogu experiments/:
#   cd hybrid_LUNAR/experiments
#   bash run_all_experiments.sh
#
# argumenty (w razie czego): lista datasetow/run_index:
#   DATASETS="CICIDS" RUN_INDICES="1" bash run_all_experiments.sh
#


set -uo pipefail

DATASETS=(${DATASETS:-CICIDS UNSW_NB15})
RUN_INDICES=(${RUN_INDICES:-1 2 3})

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

FAILED_RUNS=()

run_step() {
    local script="$1"
    local dataset="$2"
    local run_index="$3"
    local log_file="${LOG_DIR}/$(basename "$script" .py)_${dataset}_run${run_index}.log"

    echo ">>> python ${script} ${dataset} ${run_index}"
    if python "$script" "$dataset" "$run_index" > "$log_file" 2>&1; then
        echo "    OK  (log: ${log_file})"
    else
        echo "    BLAD (log: ${log_file})"
        FAILED_RUNS+=("${script} ${dataset} ${run_index}")
    fi
}

echo "=================================================================="
echo "FAZA 1: modele solo (klasyczne baseline'y + LUNAR)"
echo "=================================================================="
for dataset in "${DATASETS[@]}"; do
    for run_index in "${RUN_INDICES[@]}"; do
        run_step run_isolation_forest.py "$dataset" "$run_index"
        run_step run_lof.py "$dataset" "$run_index"
        run_step run_one_class_svm.py "$dataset" "$run_index"
        run_step run_single_experiment.py "$dataset" "$run_index"
    done
done

echo "=================================================================="
echo "FAZA 2: ensemble i analizy zalezne tylko od Fazy 1"
echo "=================================================================="
for dataset in "${DATASETS[@]}"; do
    for run_index in "${RUN_INDICES[@]}"; do
        run_step run_ensemble_without_lunar.py "$dataset" "$run_index"
        run_step run_ensemble_with_lunar.py "$dataset" "$run_index"
        run_step run_lunar_with_best_baseline.py "$dataset" "$run_index"
        run_step run_lunar_self_ensemble.py "$dataset" "$run_index"
        run_step run_lunar_epochs_analysis.py "$dataset" "$run_index"
        run_step run_lunar_fusion_strategies.py "$dataset" "$run_index"
        run_step run_robustness_analysis.py "$dataset" "$run_index"
    done
done

echo "=================================================================="
echo "FAZA 3: eksperymenty zalezne od Fazy 2 (self-ensemble)"
echo "=================================================================="
for dataset in "${DATASETS[@]}"; do
    for run_index in "${RUN_INDICES[@]}"; do
        run_step run_fusion_level_comparison.py "$dataset" "$run_index"
    done
done

echo "=================================================================="
echo "FAZA 4: agregacja wynikow"
echo "=================================================================="
if python aggregate_results.py > "${LOG_DIR}/aggregate_results.log" 2>&1; then
    echo "    OK  (log: ${LOG_DIR}/aggregate_results.log)"
else
    echo "    BLAD (log: ${LOG_DIR}/aggregate_results.log)"
    FAILED_RUNS+=("aggregate_results.py")
fi

echo "=================================================================="
echo "PODSUMOWANIE"
echo "=================================================================="
if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "Wszystkie kroki zakonczone sukcesem."
else
    echo "Zakonczono z bledami w ${#FAILED_RUNS[@]} krokach (patrz logi w ${LOG_DIR}/):"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - ${failed}"
    done
fi
