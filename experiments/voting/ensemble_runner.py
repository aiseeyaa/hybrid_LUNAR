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


def run_lunar_member(train_x, train_y, val_x, val_y, test_x, test_y,
                     dataset, seed, k, sample_type, train_new_model):
    member_name = f"{dataset}_k{k}_{sample_type}_seed{seed}"

    test_out = LUNAR.run(
        train_x, train_y, val_x, val_y, test_x, test_y,
        member_name, seed, k, sample_type, train_new_model
    )

    test_out = minmax_scale_scores(test_out)

    return {
        "name": member_name,
        "k": k,
        "sample_type": sample_type,
        "test_out": test_out
    }


def simple_weighted_soft_voting(members):
    # Na razie równe wagi, bo nie robimy osobnego przebiegu walidacyjnego
    weights = np.ones(len(members), dtype=float) / len(members)

    ensemble_test = np.zeros_like(members[0]["test_out"], dtype=float)
    for w, m in zip(weights, members):
        ensemble_test += w * m["test_out"]

    return ensemble_test, weights


def main(args):
    all_results = []

    configs = [
        {"k": 10, "sample_type": "MIXED"},
        {"k": 20, "sample_type": "MIXED"},
        {"k": 30, "sample_type": "MIXED"},
    ]

    if args.include_extra_sampling:
        configs.extend([
            {"k": 20, "sample_type": "SUBSPACE"},
            {"k": 20, "sample_type": "UNIFORM"},
        ])

    for seed in args.seeds:
        print(f"\nRunning ensemble for seed = {seed}")

        train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(args.dataset, seed)

        start = time.time()

        members = []
        for cfg in configs:
            print(f"Training member: k={cfg['k']} | samples={cfg['sample_type']}")
            member = run_lunar_member(
                train_x, train_y, val_x, val_y, test_x, test_y,
                args.dataset, seed, cfg["k"], cfg["sample_type"], args.train_new_model
            )
            members.append(member)

        ensemble_test, weights = simple_weighted_soft_voting(members)

        threshold = np.percentile(ensemble_test, 95)
        test_metrics = evaluate_scores(test_y, ensemble_test, threshold=threshold)

        runtime = time.time() - start
        test_metrics["runtime_sec"] = runtime
        all_results.append(test_metrics)

        print("\nEnsemble members:")
        for m, w in zip(members, weights):
            print(f"{m['name']} | weight={w:.4f}")

        print(f"\nThreshold (95th percentile): {threshold:.6f}")
        print_metrics(f"ENSEMBLE TEST RESULTS - seed {seed}", test_metrics)
        print(f"Runtime: {runtime:.2f} sec")

    print("\n===== ENSEMBLE MEAN RESULTS OVER SEEDS =====")
    for key in ["auc", "pr_auc", "precision", "recall", "f1", "runtime_sec"]:
        vals = [r[key] for r in all_results]
        print(f"{key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HRSS")
    parser.add_argument("--train_new_model", action="store_true")
    parser.add_argument("--include_extra_sampling", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    args = parser.parse_args()
    main(args)