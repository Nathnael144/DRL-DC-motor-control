"""
Generate report graphs/tables for the final residual SAC experiment.

Inputs:
  - training run folder with seed monitor logs and best-checkpoint CSVs
  - comparison folder from compare_sac_td3_classical.py

Outputs include training curves, best-checkpoint tables, mean +/- std tables,
winner-count charts, and per-seed selected-performance tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = SCRIPT_DIR / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_monitor_logs(run_dir: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        monitor_file = seed_dir / "monitor.csv"
        if not monitor_file.exists():
            continue
        try:
            df = pd.read_csv(monitor_file, comment="#")
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "r" not in df:
            continue
        seed = int(seed_dir.name.split("_")[-1])
        df = df.copy()
        df["seed"] = seed
        df["episode"] = np.arange(1, len(df) + 1)
        df["env_step"] = df["l"].cumsum()
        df["reward_roll10"] = df["r"].rolling(10, min_periods=1).mean()
        frames.append(df[["seed", "episode", "env_step", "r", "reward_roll10"]])
    if not frames:
        raise FileNotFoundError(f"No monitor logs found under {run_dir}")
    return pd.concat(frames, ignore_index=True)


def save_table_image(df: pd.DataFrame, output_path: Path, title: str, font_size: float = 8.0) -> None:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")

    fig_h = max(2.4, 0.38 * len(display) + 1.2)
    fig_w = max(9.0, 1.45 * len(display.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.25)
    ax.set_title(title, pad=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(monitor_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    for seed, group in monitor_df.groupby("seed"):
        ax.plot(group["env_step"], group["r"], alpha=0.25, linewidth=0.8)
        ax.plot(group["env_step"], group["reward_roll10"], linewidth=1.4, label=f"Seed {seed:02d}")
    ax.set_title("Residual SAC Training Curves Across 10 Seeds")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_learning_curves_all_seeds.png")
    plt.close(fig)

    pivot = monitor_df.pivot_table(index="episode", columns="seed", values="reward_roll10")
    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.plot(mean.index, mean.values, color="#1f6f8b", linewidth=2.2, label="Mean rolling reward")
    ax.fill_between(mean.index, mean - std, mean + std, color="#1f6f8b", alpha=0.18, label="+/- 1 std")
    ax.set_title("Mean Training Reward Across Seeds")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling reward, window=10")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_mean_reward_with_std.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    pivot = monitor_df.pivot_table(index="env_step", columns="seed", values="reward_roll10")
    mean_by_step = pivot.mean(axis=1).sort_index()
    std_by_step = pivot.std(axis=1).reindex(mean_by_step.index)
    ax.plot(mean_by_step.index, mean_by_step.values, color="#214E34", linewidth=2.2)
    ax.fill_between(
        mean_by_step.index,
        mean_by_step - std_by_step,
        mean_by_step + std_by_step,
        color="#214E34",
        alpha=0.16,
    )
    stages = [
        (0, 25_000, "Nominal\nuniform", "#EAF4F4"),
        (25_000, 50_000, "Robust\nuniform", "#F6F2D4"),
        (50_000, 75_000, "Hard\nuniform", "#F7DAD9"),
        (75_000, 100_000, "Hard\ntracking", "#E9D8FD"),
    ]
    ymin, ymax = ax.get_ylim()
    for start, end, label, color in stages:
        ax.axvspan(start, end, color=color, alpha=0.45, zorder=0)
        ax.text(
            (start + end) / 2,
            ymax - 0.08 * (ymax - ymin),
            label,
            ha="center",
            va="top",
            fontsize=9,
            color="#222222",
        )
    for boundary in [25_000, 50_000, 75_000]:
        ax.axvline(boundary, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Training Reward With Curriculum Stages")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean rolling reward, window=10")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_reward_with_curriculum_stages.png")
    plt.close(fig)


def make_training_reward_summary(monitor_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for seed, group in monitor_df.groupby("seed"):
        rows.append(
            {
                "seed": seed,
                "episodes": len(group),
                "final_reward": float(group["r"].iloc[-1]),
                "recent10_mean_reward": float(group["r"].tail(10).mean()),
                "best_roll10_reward": float(group["reward_roll10"].max()),
                "final_env_step": int(group["env_step"].iloc[-1]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("seed")
    summary.to_csv(output_dir / "training_reward_summary.csv", index=False)
    save_table_image(
        summary,
        output_dir / "training_reward_summary_table.png",
        "Training Reward Summary by Seed",
    )
    return summary


def make_curriculum_stage_summary(monitor_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    def stage_for_step(step: float) -> str:
        if step <= 25_000:
            return "1_nominal_uniform"
        if step <= 50_000:
            return "2_robust_uniform"
        if step <= 75_000:
            return "3_hard_uniform"
        return "4_hard_tracking"

    staged = monitor_df.copy()
    staged["stage"] = staged["env_step"].map(stage_for_step)
    summary = (
        staged.groupby("stage")["r"]
        .agg(episodes="count", mean_reward="mean", std_reward="std", min_reward="min", max_reward="max")
        .reset_index()
    )
    summary.to_csv(output_dir / "curriculum_stage_reward_summary.csv", index=False)
    save_table_image(
        summary,
        output_dir / "curriculum_stage_reward_summary_table.png",
        "Reward Summary by Curriculum Stage",
    )
    return summary


def make_best_checkpoint_table(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "best_checkpoint_by_seed.csv")
    cols = ["seed", "checkpoint", "IAE", "SSE", "ControlEnergy", "SaturationFraction", "objective"]
    table_df = df[cols].copy()
    table_df.to_csv(output_dir / "best_checkpoint_table.csv", index=False)
    save_table_image(
        table_df,
        output_dir / "best_checkpoint_table.png",
        "Selected Best Checkpoint by Seed",
    )
    return table_df


def format_mean_std(row: pd.Series, metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    seed_std = row.get(f"{metric}_std_across_seeds")
    if pd.isna(seed_std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {seed_std:.4f}"


def make_overall_summary_table(comparison_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(comparison_dir / "overall_mean_std_summary.csv")
    table = pd.DataFrame(
        {
            "controller": df["controller"],
            "family": df["family"],
            "n_seeds": df["n_seeds"],
            "IAE": [format_mean_std(row, "IAE") for _, row in df.iterrows()],
            "ControlEnergy": [format_mean_std(row, "ControlEnergy") for _, row in df.iterrows()],
            "SSE": [format_mean_std(row, "SSE") for _, row in df.iterrows()],
            "SaturationFraction": [format_mean_std(row, "SaturationFraction") for _, row in df.iterrows()],
        }
    )
    table.to_csv(output_dir / "overall_mean_std_report_table.csv", index=False)
    save_table_image(
        table,
        output_dir / "overall_mean_std_report_table.png",
        "Overall Mean +/- Std Comparison",
        font_size=7.6,
    )
    return table


def make_hard_summary_table(comparison_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(comparison_dir / "hard_cases_mean_std_summary.csv")
    table = pd.DataFrame(
        {
            "controller": df["controller"],
            "family": df["family"],
            "hard_IAE": [format_mean_std(row, "IAE") for _, row in df.iterrows()],
            "hard_ControlEnergy": [format_mean_std(row, "ControlEnergy") for _, row in df.iterrows()],
            "hard_SSE": [format_mean_std(row, "SSE") for _, row in df.iterrows()],
            "hard_SaturationFraction": [format_mean_std(row, "SaturationFraction") for _, row in df.iterrows()],
        }
    )
    table.to_csv(output_dir / "hard_cases_mean_std_report_table.csv", index=False)
    save_table_image(
        table,
        output_dir / "hard_cases_mean_std_report_table.png",
        "Hard Nonlinear Cases Mean +/- Std",
        font_size=7.6,
    )
    return table


def make_selected_seed_table(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    metrics_file = run_dir / "best_selected" / "all_seed_metrics_long.csv"
    df = pd.read_csv(metrics_file)
    table = (
        df.groupby("seed")[["IAE", "SSE", "ControlEnergy", "SaturationFraction"]]
        .mean()
        .reset_index()
        .sort_values("seed")
    )
    table.to_csv(output_dir / "selected_seed_mean_metrics.csv", index=False)
    save_table_image(
        table,
        output_dir / "selected_seed_mean_metrics_table.png",
        "Best-Selected Residual SAC Mean Metrics by Seed",
    )
    return table


def plot_seed_metric_boxplots(seed_table: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["IAE", "SSE", "ControlEnergy", "SaturationFraction"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), dpi=180)
    for ax, metric in zip(axes, metrics):
        ax.boxplot(seed_table[metric].to_numpy(), tick_labels=[metric])
        ax.scatter(np.ones(len(seed_table)), seed_table[metric], alpha=0.75, s=22)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Seed Variability for Best-Selected Residual SAC")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_seed_metric_boxplots.png")
    plt.close(fig)


def plot_winner_counts(comparison_dir: Path, output_dir: Path) -> pd.DataFrame:
    winners = pd.read_csv(comparison_dir / "winner_table_by_case.csv")
    rows = []
    for metric, column in [
        ("IAE", "best_iae_controller"),
        ("SSE", "best_sse_controller"),
        ("Energy", "best_energy_controller"),
    ]:
        counts = winners[column].value_counts()
        for controller, count in counts.items():
            rows.append({"metric": metric, "controller": controller, "wins": int(count)})
    count_df = pd.DataFrame(rows)
    count_df.to_csv(output_dir / "winner_count_table.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=180)
    for ax, metric in zip(axes, ["IAE", "SSE", "Energy"]):
        subset = count_df[count_df["metric"] == metric].sort_values("wins", ascending=False)
        ax.bar(subset["controller"], subset["wins"], color="#2E86AB")
        ax.set_title(f"{metric} wins")
        ax.set_ylabel("Cases won")
        ax.set_ylim(0, 24)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Controller Winner Counts Across 24 Cases")
    fig.tight_layout()
    fig.savefig(output_dir / "winner_count_bars.png")
    plt.close(fig)

    save_table_image(
        count_df,
        output_dir / "winner_count_table.png",
        "Winner Counts Across 24 Test Cases",
    )
    return count_df


def parse_args() -> argparse.Namespace:
    default_run_dir = (
        SCRIPT_DIR
        / "outputs"
        / "residual_sac_curriculum_mpcwarm_10seeds_100k"
    )
    default_comparison_dir = (
        SCRIPT_DIR
        / "outputs"
        / "residual_sac_curriculum_mpcwarm_10seeds_100k_best_comparison"
    )
    default_output_dir = (
        SCRIPT_DIR
        / "outputs"
        / "residual_sac_curriculum_mpcwarm_10seeds_100k_report_graphs"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--comparison-dir", type=Path, default=default_comparison_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    monitor_df = read_monitor_logs(args.run_dir)
    monitor_df.to_csv(args.output_dir / "training_monitor_long.csv", index=False)
    plot_training_curves(monitor_df, args.output_dir)
    make_training_reward_summary(monitor_df, args.output_dir)
    make_curriculum_stage_summary(monitor_df, args.output_dir)
    make_best_checkpoint_table(args.run_dir, args.output_dir)
    make_overall_summary_table(args.comparison_dir, args.output_dir)
    make_hard_summary_table(args.comparison_dir, args.output_dir)
    seed_table = make_selected_seed_table(args.run_dir, args.output_dir)
    plot_seed_metric_boxplots(seed_table, args.output_dir)
    plot_winner_counts(args.comparison_dir, args.output_dir)

    manifest = pd.DataFrame(
        {
            "file": sorted(path.name for path in args.output_dir.iterdir() if path.is_file())
        }
    )
    manifest.to_csv(args.output_dir / "report_manifest.csv", index=False)
    print(f"[done] report graphs/tables written to: {args.output_dir}", flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
