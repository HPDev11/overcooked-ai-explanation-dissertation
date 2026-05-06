import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = [
    "total_game_reward",
    "total_shaped_reward",
    "idle_rate_avg",
    "collisions",
    "blocked_partner_total",
    "blocked_wall_total",
    "handoff_total",
    "exp_any_rate",
    "exp_any_count",
    "soups_delivered",
    "time_to_first_delivery",
    "mean_delivery_interval",
    "productive_actions_0",
    "productive_actions_1",
    "productive_actions_total",
    "contribution_ratio_0",
    "contribution_ratio_1",
    "activity_count_0",
    "activity_count_1",
    "activity_0",
    "activity_1",
    "interactivity_count_0",
    "interactivity_count_1",
    "interactivity_0",
    "interactivity_1",
    "efficiency_0",
    "efficiency_1",
    "adjacency_rate",
    "plate_pickup_total",
    "plate_pickup_while_cooking_count",
    "plate_pickup_while_cooking_rate",
    "pot_empty_rate",
    "pot_cooking_rate",
    "pot_ready_wait_rate",
]

GROUP_KEYS = [
    "layout",
    "condition",
    "mode",
    "goal_change_only",
    "event_trigger",
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sample_std(xs):
    n = len(xs)
    if n <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def ci95(xs):
    n = len(xs)
    if n <= 1:
        return 0.0
    return 1.96 * sample_std(xs) / math.sqrt(n)


def load_summary_jsons(input_dirs):
    records = []
    for d in input_dirs:
        dpath = Path(d)
        if not dpath.exists():
            print(f"[WARN] Input dir not found: {dpath}")
            continue

        files = sorted(dpath.glob("*_summary.json"))
        if not files:
            print(f"[WARN] No summary JSON files found in: {dpath}")
            continue

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                rec["_source_file"] = str(fp)
                records.append(rec)
            except Exception as e:
                print(f"[WARN] Failed to read {fp}: {e}")
    return records


def group_records(records):
    grouped = defaultdict(list)
    for r in records:
        key = tuple(r.get(k) for k in GROUP_KEYS)
        grouped[key].append(r)
    return grouped


def aggregate_group(group_key, rows):
    out = {}
    for k, v in zip(GROUP_KEYS, group_key):
        out[k] = v

    out["n_runs"] = len(rows)

    for metric in METRICS:
        vals = [float(r.get(metric, 0.0)) for r in rows]
        out[f"{metric}_mean"] = mean(vals)
        out[f"{metric}_std"] = sample_std(vals)
        out[f"{metric}_ci95"] = ci95(vals)

    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] No rows to write for {path}")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved CSV: {path}")


def try_write_xlsx(path, rows):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("[INFO] openpyxl not installed; skipping XLSX output.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "results"

    headers = list(rows[0].keys())
    ws.append(headers)

    for row in rows:
        ws.append([row.get(h) for h in headers])

    wb.save(path)
    print(f"Saved XLSX: {path}")


def write_run_level_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        print(f"[WARN] No run-level records to write for {path}")
        return

    keys = set()
    for r in records:
        keys.update(r.keys())

    fieldnames = sorted(k for k in keys if not k.startswith("_"))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Saved run-level CSV: {path}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate simulator summary JSON results.")
    ap.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="One or more directories containing *_summary.json files",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for aggregated tables",
    )
    ap.add_argument(
        "--write_xlsx",
        action="store_true",
        help="Also write XLSX if openpyxl is available",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_summary_jsons(args.input_dirs)
    if not records:
        print("[ERROR] No summary records found.")
        return

    grouped = group_records(records)

    agg_rows = []
    for group_key, rows in grouped.items():
        agg_rows.append(aggregate_group(group_key, rows))

    agg_rows.sort(
        key=lambda r: (
            str(r.get("layout")),
            str(r.get("mode")),
            str(r.get("condition")),
            str(r.get("goal_change_only")),
            str(r.get("event_trigger")),
        )
    )

    write_run_level_csv(out_dir / "run_level_results.csv", records)
    write_csv(out_dir / "results_table.csv", agg_rows)

    if args.write_xlsx:
        try_write_xlsx(out_dir / "results_table.xlsx", agg_rows)

    print("\nDone.")
    print("Main output:")
    print(f"  {out_dir / 'results_table.csv'}")


if __name__ == "__main__":
    main()
