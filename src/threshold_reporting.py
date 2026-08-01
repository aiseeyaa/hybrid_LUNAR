"""
Shared helper for the three-threshold-variant reporting protocol requested
in the review (31.07.2026 feedback):

  1. "standard"         -> maximize F1 (as before)
  2. "recall_oriented"  -> maximize F-beta with beta>1 (favors recall)
  3. "operational"      -> fix a minimum acceptable recall, then maximize precision

All three thresholds are selected exclusively on the calibration split
(val_calib, returned by data.make_final_subsample) and then frozen and
applied, unmodified, to the held-out test set. This keeps threshold
selection and final evaluation on disjoint data, and applies an identical
protocol across every model_type so comparisons between LUNAR, the solo
baselines, and the ensembles are apples-to-apples.

This module intentionally does not touch LUNAR.py, utils.py or
variables.py -- it only orchestrates calls into metrics.py.
"""

from metrics import (
    compare_threshold_strategies,
    evaluate_scores,
    pr_curve_summary,
)


def calibrate_and_evaluate(val_y, scores_val, test_y, scores_test,
                            beta=2.0, normal_q=0.99, min_recall=0.90,
                            n_pr_points=50):
    """
    Runs threshold calibration on (val_y, scores_val) and evaluates all
    three reporting variants on the frozen (test_y, scores_test).

    Returns:
        metrics_standard      : dict from evaluate_scores at the F1-max threshold
                                 (this is what "metrics" / build_experiment_record
                                 expects as its primary metrics argument)
        threshold_info         : dict describing how metrics_standard's threshold
                                  was chosen (same shape as before, for
                                  backward-compatible top-level "threshold" field)
        threshold_variants      : dict with "standard" / "recall_oriented" /
                                  "operational" -> each an evaluate_scores() dict
                                  on the test set
        pr_curve                : list of {threshold, precision, recall, f1} rows
                                  computed on the calibration split
    """
    threshold_candidates = compare_threshold_strategies(
        val_y, scores_val, beta=beta, normal_q=normal_q, min_recall=min_recall
    )

    standard_threshold = threshold_candidates["f1_max"]["threshold"]
    recall_oriented_threshold = threshold_candidates[f"f{beta:g}_max"]["threshold"]
    operational_threshold = threshold_candidates["operational"]["threshold"]

    metrics_standard = evaluate_scores(test_y, scores_test, threshold=standard_threshold)
    metrics_recall_oriented = evaluate_scores(test_y, scores_test, threshold=recall_oriented_threshold)
    metrics_operational = evaluate_scores(test_y, scores_test, threshold=operational_threshold)

    threshold_info = {
        "selection_method": "validation_f1_max",
        **threshold_candidates["f1_max"],
    }

    threshold_variants = {
        "standard": metrics_standard,
        "recall_oriented": metrics_recall_oriented,
        "operational": metrics_operational,
    }

    pr_curve = pr_curve_summary(val_y, scores_val, n_points=n_pr_points)

    return metrics_standard, threshold_info, threshold_variants, pr_curve, threshold_candidates
