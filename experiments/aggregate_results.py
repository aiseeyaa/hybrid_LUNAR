import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "seed_81"
AGG_DIR = RESULTS_DIR / "aggregation" / "seed_81"


def load_records(results_dir):
    json_files = sorted(Path(results_dir).glob("*.json"))
    records = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print("SKIP", path.name, e)
    return records


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def cm_to_dict(cm, prefix="ConfusionMatrix"):
    """Rozbija [[TN, FP], [FN, TP]] na 4 kolumny skalarne."""
    if not cm:
        return {}
    (tn, fp), (fn, tp) = cm[0], cm[1]
    return {f"{prefix}_TN": tn, f"{prefix}_FP": fp, f"{prefix}_FN": fn, f"{prefix}_TP": tp}


def flatten_threshold_variants(records):
    rows = []
    for rec in records:
        rec = dict(rec)  # nie mutujemy oryginalu
        pr_curve = rec.pop("pr_curve", None)
        variants = rec.pop("threshold_variants", None)
        cm_top = rec.pop("ConfusionMatrix", None)
        hyperparams = rec.pop("hyperparameters", None)
        threshold_info = rec.pop("threshold_info", None)

        base = dict(rec)
        if hyperparams:
            base.update(flatten_dict(hyperparams, "hyperparameters"))
        if threshold_info:
            base.update(flatten_dict(threshold_info, "threshold_info"))
        if cm_top:
            base.update(cm_to_dict(cm_top))
        if pr_curve:
            base["pr_curve_json"] = json.dumps(pr_curve)
            base["pr_curve_n_points"] = len(pr_curve)

        if not variants:
            base["threshold_variant"] = "standard"
            rows.append(base)
            continue

        for variant_name, variant_metrics in variants.items():
            row = dict(base)
            row["threshold_variant"] = variant_name
            vm = dict(variant_metrics)
            vcm = vm.pop("ConfusionMatrix", None)
            row.update(vm)  # WSZYSTKIE pola wariantu (nie tylko wybrane metryki)
            if vcm:
                row.update(cm_to_dict(vcm))
            rows.append(row)
    return rows


def build_pr_curves_long(records):
    rows = []
    for rec in records:
        pr_curve = rec.get("pr_curve")
        if not pr_curve:
            continue
        for pt in pr_curve:
            rows.append({
                "experiment_id": rec.get("experiment_id"),
                "dataset_name": rec.get("dataset_name"),
                "model_type": rec.get("model_type"),
                "fusion_strategy": rec.get("fusion_strategy"),
                "curve_threshold": pt.get("threshold"),
                "precision": pt.get("precision"),
                "recall": pt.get("recall"),
                "f1": pt.get("f1"),
            })
    return pd.DataFrame(rows)


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
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    })

    for metric in [m for m in ["AUC_ROC", "AUC_PR", "F1", "runtime_train"] if m in df.columns]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="model_type", y=metric, hue="dataset_name",
                    palette="Set2", showfliers=False)
        sns.stripplot(data=df, x="model_type", y=metric, hue="dataset_name",
              dodge=True, jitter=0.2, palette="dark:black", alpha=0.5, size=4, legend=False)
        plt.title(f"{metric} by Model Type")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(agg_dir / f"box_{metric}_by_model.png", dpi=300, bbox_inches="tight")
        plt.savefig(agg_dir / f"box_{metric}_by_model.pdf", bbox_inches="tight")
        plt.close()

    if "threshold_variant" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="threshold_variant", y="Recall", hue="model_type",
                    palette="Set2", showfliers=False)
        sns.stripplot(data=df, x="model_type", y=metric, hue="dataset_name",
              dodge=True, jitter=0.2, palette="dark:black", alpha=0.5, size=4, legend=False)
        plt.title("Recall by Threshold Variant")
        plt.tight_layout()
        plt.savefig(agg_dir / "box_recall_by_threshold_variant.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="threshold_variant", y="Precision", hue="model_type",
                    palette="Set2", showfliers=False)
        sns.stripplot(data=df, x="model_type", y=metric, hue="dataset_name",
              dodge=True, jitter=0.2, palette="dark:black", alpha=0.5, size=4, legend=False)
        plt.title("Precision by Threshold Variant")
        plt.tight_layout()
        plt.savefig(agg_dir / "box_precision_by_threshold_variant.png", dpi=300, bbox_inches="tight")
        plt.savefig(agg_dir / "box_precision_by_threshold_variant.pdf", bbox_inches="tight")
        plt.close()

    if "fusion_strategy" in df.columns:
        df_fusion = df.dropna(subset=["fusion_strategy"])
        if not df_fusion.empty:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df_fusion, x="fusion_strategy", y="F1", hue="dataset_name",
                        palette="Set2", showfliers=False)
            sns.stripplot(data=df, x="model_type", y=metric, hue="dataset_name",
              dodge=True, jitter=0.2, palette="dark:black", alpha=0.5, size=4, legend=False)
            plt.title("F1 by Fusion Strategy")
            plt.tight_layout()
            plt.savefig(agg_dir / "box_f1_by_fusion.png", dpi=300, bbox_inches="tight")
            plt.savefig(agg_dir / "box_f1_by_fusion.pdf", bbox_inches="tight")
            plt.close()


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

    pr_long = build_pr_curves_long(records)
    if not pr_long.empty:
        pr_long.to_csv(agg_dir / "pr_curves_long.csv", index=False)

    print(f"Aggregated {len(records)} records ({len(df)} rows incl. threshold variants, "
          f"{len(df.columns)} columns) into {agg_dir}")
    if not pr_long.empty:
        print(f"Zapisano dodatkowo pr_curves_long.csv ({len(pr_long)} punktow krzywych PR)")


if __name__ == "__main__":
    main()
