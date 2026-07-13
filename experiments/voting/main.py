import sys
from pathlib import Path
import time
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.LUNAR import LUNAR
from external.LUNAR import utils
from metrics import minmax_scale_scores, evaluate_scores, print_metrics


def run_single_lunar(train_x, train_y, val_x, val_y, test_x, test_y,
                     dataset, seed, k, sample_type, train_new_model):
    test_out = LUNAR.run(
        train_x, train_y, val_x, val_y, test_x, test_y,
        dataset, seed, k, sample_type, train_new_model
    )

    test_out = minmax_scale_scores(test_out)

    threshold = np.percentile(test_out, 95)
    test_metrics = evaluate_scores(test_y, test_out, threshold=threshold)

    return test_out, threshold, test_metrics


def main(args):
    all_results = []

    for seed in args.seeds:
        print(f"\nRunning trial with random seed = {seed}")

        train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(args.dataset, seed)

        start = time.time()

        _, threshold, test_metrics = run_single_lunar(
            train_x, train_y, val_x, val_y, test_x, test_y,
            args.dataset, seed, args.k, args.samples, args.train_new_model
        )

        runtime = time.time() - start
        test_metrics["runtime_sec"] = runtime
        all_results.append(test_metrics)

        print(f"Dataset: {args.dataset} | Samples: {args.samples} | k: {args.k}")
        print(f"Threshold (95th percentile): {threshold:.6f}")
        print_metrics(f"TEST RESULTS - seed {seed}", test_metrics)
        print(f"Runtime: {runtime:.2f} sec")

    print("\n===== MEAN RESULTS OVER SEEDS =====")
    for key in ["auc", "pr_auc", "precision", "recall", "f1", "runtime_sec"]:
        vals = [r[key] for r in all_results]
        print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HRSS")
    parser.add_argument("--samples", type=str, default="MIXED")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--train_new_model", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    args = parser.parse_args()
    main(args)