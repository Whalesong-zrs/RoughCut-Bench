import argparse
import csv
import os
from statistics import mean, stdev


METRIC_NAMES = ["precision", "recall", "f1", "avg_iou", "kendalltau"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize repeat eval CSVs across test ids for one model/chain/version."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--chain", required=True, choices=["unified_workflow", "visual_workflow", "joint_video_understanding_workflow"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--version-root", required=True)
    parser.add_argument("--test-ids", nargs="+", required=True)
    parser.add_argument("--set-names", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def eval_csv_name(chain, set_name, model):
    if chain == "unified_workflow":
        return f"eval_unified_cut_plans_{set_name}_{model}.csv"
    return f"eval_cut_plans_{set_name}_{model}.csv"


def read_metrics(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    if len(rows) < 2:
        raise ValueError(f"expected at least 2 rows in {csv_path}")

    header = [name.strip() for name in rows[0]]
    values = [value.strip() for value in rows[1]]
    row = dict(zip(header, values))

    if "f1" not in row and "precision" in row and "recall" in row:
        precision = float(row["precision"])
        recall = float(row["recall"])
        row["f1"] = str(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    missing = [name for name in METRIC_NAMES if name not in row]
    if missing:
        raise KeyError(f"missing columns {missing} in {csv_path}")
    return {name: float(row[name]) for name in METRIC_NAMES}


def fmt(value):
    text = f"{value:.6f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def main():
    args = parse_args()
    output = args.output or os.path.join(args.version_root, "summary.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    rows = []
    for set_name in args.set_names:
        by_test = []
        for test_id in args.test_ids:
            csv_path = os.path.join(
                args.version_root,
                test_id,
                set_name,
                eval_csv_name(args.chain, set_name, args.model),
            )
            metrics = read_metrics(csv_path)
            by_test.append(metrics)
            rows.append({
                "model": args.model,
                "chain": args.chain,
                "version": args.version,
                "set_name": set_name,
                "stat": test_id,
                **{name: fmt(metrics[name]) for name in METRIC_NAMES},
            })

        mean_metrics = {
            name: mean(metrics[name] for metrics in by_test)
            for name in METRIC_NAMES
        }
        std_metrics = {
            name: stdev(metrics[name] for metrics in by_test)
            for name in METRIC_NAMES
        }
        rows.append({
            "model": args.model,
            "chain": args.chain,
            "version": args.version,
            "set_name": set_name,
            "stat": "mean",
            **{name: fmt(mean_metrics[name]) for name in METRIC_NAMES},
        })
        rows.append({
            "model": args.model,
            "chain": args.chain,
            "version": args.version,
            "set_name": set_name,
            "stat": "std",
            **{name: fmt(std_metrics[name]) for name in METRIC_NAMES},
        })

    fieldnames = ["model", "chain", "version", "set_name", "stat"] + METRIC_NAMES
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
