import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

PLOT_METRICS = [
    ("total_game_reward_mean", "total_game_reward_ci95", "Total game reward"),
    ("total_shaped_reward_mean", "total_shaped_reward_ci95", "Total shaped reward"),
    ("soups_delivered_mean", "soups_delivered_ci95", "Soups delivered"),
    ("time_to_first_delivery_mean", "time_to_first_delivery_ci95", "Time to first delivery"),
    ("mean_delivery_interval_mean", "mean_delivery_interval_ci95", "Mean delivery interval"),
    ("idle_rate_avg_mean", "idle_rate_avg_ci95", "Idle rate (avg)"),
    ("collisions_mean", "collisions_ci95", "Collisions"),
    ("blocked_partner_total_mean", "blocked_partner_total_ci95", "Blocked partner total"),
    ("blocked_wall_total_mean", "blocked_wall_total_ci95", "Blocked wall total"),
    ("productive_actions_0_mean", "productive_actions_0_ci95", "Productive actions (agent 0)"),
    ("productive_actions_1_mean", "productive_actions_1_ci95", "Productive actions (agent 1)"),
    ("productive_actions_total_mean", "productive_actions_total_ci95", "Productive actions (total)"),
    ("contribution_ratio_0_mean", "contribution_ratio_0_ci95", "Contribution ratio (agent 0)"),
    ("contribution_ratio_1_mean", "contribution_ratio_1_ci95", "Contribution ratio (agent 1)"),
    ("activity_0_mean", "activity_0_ci95", "Activity rate (agent 0)"),
    ("activity_1_mean", "activity_1_ci95", "Activity rate (agent 1)"),
    ("interactivity_0_mean", "interactivity_0_ci95", "Interactivity rate (agent 0)"),
    ("interactivity_1_mean", "interactivity_1_ci95", "Interactivity rate (agent 1)"),
    ("efficiency_0_mean", "efficiency_0_ci95", "Efficiency (agent 0)"),
    ("efficiency_1_mean", "efficiency_1_ci95", "Efficiency (agent 1)"),
    ("adjacency_rate_mean", "adjacency_rate_ci95", "Adjacency rate"),
    ("plate_pickup_while_cooking_rate_mean", "plate_pickup_while_cooking_rate_ci95", "Plate pickup while cooking rate"),
    ("pot_empty_rate_mean", "pot_empty_rate_ci95", "Pot empty rate"),
    ("pot_cooking_rate_mean", "pot_cooking_rate_ci95", "Pot cooking rate"),
    ("pot_ready_wait_rate_mean", "pot_ready_wait_rate_ci95", "Pot ready-wait rate"),
    ("exp_any_rate_mean", "exp_any_rate_ci95", "Explanation rate"),
]


def load_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def filter_rows(rows, layout=None, mode=None, goal_change_only=None, event_trigger=None):
    out = []
    for r in rows:
        if layout is not None and r.get("layout") != layout:
            continue
        if mode is not None and r.get("mode") != mode:
            continue
        if goal_change_only is not None and str(r.get("goal_change_only")) != str(goal_change_only):
            continue
        if event_trigger is not None and str(r.get("event_trigger")) != str(event_trigger):
            continue
        out.append(r)
    return out


def save_bar_plot(rows, metric_col, ci_col, title, out_path):
    rows = sorted(rows, key=lambda r: r.get("condition", ""))

    labels = [r.get("condition", "") for r in rows]
    values = [as_float(r.get(metric_col, 0.0)) for r in rows]
    errors = [as_float(r.get(ci_col, 0.0)) for r in rows]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, values, yerr=errors, capsize=6)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xlabel("Condition")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def save_scatter_plot(run_rows, metric, title, out_path):
    exp_vals = []
    noexp_vals = []

    for r in run_rows:
        cond = r.get("condition", "")
        val = as_float(r.get(metric, 0.0))
        if cond == "exp":
            exp_vals.append(val)
        elif cond == "noexp":
            noexp_vals.append(val)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    if noexp_vals:
        ax.scatter([0] * len(noexp_vals), noexp_vals, label="noexp")
    if exp_vals:
        ax.scatter([1] * len(exp_vals), exp_vals, label="exp")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["noexp", "exp"])
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xlabel("Condition")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot aggregated simulator results.")
    ap.add_argument(
        "--results_csv",
        type=str,
        default="results/results_table.csv",
        help="Aggregated CSV from aggregate_results.py",
    )
    ap.add_argument(
        "--run_level_csv",
        type=str,
        default="results/run_level_results.csv",
        help="Run-level CSV from aggregate_results.py",
    )
    ap.add_argument("--layout", type=str, required=True)
    ap.add_argument("--mode", type=str, default="both")
    ap.add_argument("--goal_change_only", type=str, default="True")
    ap.add_argument("--event_trigger", type=str, default="True")
    ap.add_argument("--fig_dir", type=str, default="figures")
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    agg_rows = load_csv(args.results_csv)
    run_rows = load_csv(args.run_level_csv)

    subset = filter_rows(
        agg_rows,
        layout=args.layout,
        mode=args.mode,
        goal_change_only=args.goal_change_only,
        event_trigger=args.event_trigger,
    )

    if not subset:
        print("[ERROR] No matching rows found in aggregated results.")
        return

    for metric_col, ci_col, title in PLOT_METRICS:
        out_name = f"{args.layout}_{args.mode}_{metric_col}.png"
        save_bar_plot(subset, metric_col, ci_col, title, fig_dir / out_name)

    run_subset = filter_rows(
        run_rows,
        layout=args.layout,
        mode=args.mode,
        goal_change_only=args.goal_change_only,
        event_trigger=args.event_trigger,
    )

    if run_subset:
        save_scatter_plot(
            run_subset,
            "total_game_reward",
            "Per-run total game reward",
            fig_dir / f"{args.layout}_{args.mode}_scatter_total_game_reward.png",
        )
        save_scatter_plot(
            run_subset,
            "idle_rate_avg",
            "Per-run idle rate (avg)",
            fig_dir / f"{args.layout}_{args.mode}_scatter_idle_rate_avg.png",
        )
        save_scatter_plot(
            run_subset,
            "soups_delivered",
            "Per-run soups delivered",
            fig_dir / f"{args.layout}_{args.mode}_scatter_soups_delivered.png",
        )
        save_scatter_plot(
            run_subset,
            "time_to_first_delivery",
            "Per-run time to first delivery",
            fig_dir / f"{args.layout}_{args.mode}_scatter_time_to_first_delivery.png",
        )
        save_scatter_plot(
            run_subset,
            "productive_actions_total",
            "Per-run productive actions total",
            fig_dir / f"{args.layout}_{args.mode}_scatter_productive_actions_total.png",
        )
        save_scatter_plot(
            run_subset,
            "pot_ready_wait_rate",
            "Per-run pot ready-wait rate",
            fig_dir / f"{args.layout}_{args.mode}_scatter_pot_ready_wait_rate.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
