import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
AGG_DIR = RESULTS_DIR / "aggregation"


def load_records(results_dir):
    json_files = sorted(results_dir.glob("*.json"))
    records = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print("SKIP", path.name, e)
    return records


def flatten_threshold_variants(records):
    """
    Expands each record's threshold_variants (standard / recall_oriented /
    operational) into separate rows tagged by "threshold_variant", so the
    aggregation and plots can show model comparisons across all three
    reporting variants requested in the review, not just the default
    (F1-max) one.
    """
    rows = []
    for rec in records:
        base = {k: v for k, v in rec.items() if k not in ("threshold_variants", "pr_curve", "ConfusionMatrix")}
        variants = rec.get("threshold_variants")
        if not variants:
            base["threshold_variant"] = "standard"
            rows.append(base)
            continue
        for variant_name, variant_metrics in variants.items():
            row = dict(base)
            row["threshold_variant"] = variant_name
            for m in ("AUC_ROC", "AUC_PR", "Precision", "Recall", "F1", "threshold"):
                if m in variant_metrics:
                    row[m] = variant_metrics[m]
            rows.append(row)
    return rows


def build_summary(df, agg_dir):
    agg_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(agg_dir / "all_results_flat.csv", index=False)

    metric_cols = [c for c in ["AUC_ROC", "AUC_PR", "Precision", "Recall", "F1",
                                "runtime_train", "runtime_inference"] if c in df.columns]
    group_cols = [c for c in ["dataset_name", "model_type", "fusion_strategy", "threshold_variant"]
                  if c in df.columns]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary.to_csv(agg_dir / "summary_by_dataset_model_fusion_variant.csv")
    return summary


def build_plots(df, agg_dir):
    for metric in [m for m in ["AUC_ROC", "AUC_PR", "F1", "runtime_train"] if m in df.columns]:
        fig = px.box(df, x="model_type", y=metric, color="dataset_name", points="all",
                     title=f"{metric} by model_type")
        fig.write_html(agg_dir / f"box_{metric}_by_model.html")

    if "threshold_variant" in df.columns:
        fig = px.box(df, x="threshold_variant", y="Recall", color="model_type", points="all",
                     title="Recall by threshold variant (standard / recall_oriented / operational)")
        fig.write_html(agg_dir / "box_recall_by_threshold_variant.html")

        fig2 = px.box(df, x="threshold_variant", y="Precision", color="model_type", points="all",
                      title="Precision by threshold variant (standard / recall_oriented / operational)")
        fig2.write_html(agg_dir / "box_precision_by_threshold_variant.html")

    if "fusion_strategy" in df.columns:
        fig = px.box(df.dropna(subset=["fusion_strategy"]), x="fusion_strategy", y="F1",
                     color="dataset_name", points="all", title="F1 by fusion strategy")
        fig.write_html(agg_dir / "box_f1_by_fusion.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    agg_dir = results_dir / "aggregation"

    records = load_records(results_dir)
    rows = flatten_threshold_variants(records)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No JSON result files found in {results_dir}")

    build_summary(df, agg_dir)
    build_plots(df, agg_dir)

    print(f"Aggregated {len(records)} records ({len(df)} rows incl. threshold variants) into {agg_dir}")


if __name__ == "__main__":
    main()
