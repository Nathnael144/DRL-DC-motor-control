from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent / "outputs" / ".mplcache_graphs").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    folder: str


EXPERIMENTS = (
    ExperimentSpec("Experiment 1: tracking-focused", "experiment1_v2_10seeds_200k"),
    ExperimentSpec("Experiment 2: strict energy", "experiment2_hard_energy_10seeds_50k"),
    ExperimentSpec("Experiment 3: balanced energy", "experiment3_balanced_energy_10seeds_50k"),
    ExperimentSpec("Experiment 4: condition-aware", "experiment4_condition_aware_10seeds_100k"),
)

SCENARIO_ORDER = ("step_nominal", "step_load_disturbance", "ramp", "sine")
HARD_CONDITIONS = {"saturation", "combined_stress"}
HARD_SCENARIOS = {"step_nominal", "step_load_disturbance", "sine"}


def read_metrics(experiment_dir: Path) -> pd.DataFrame:
    frames = []
    for file_path in sorted(experiment_dir.glob("seed_*/seed_*_metrics.csv")):
        frames.append(pd.read_csv(file_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_rewards(experiment_dir: Path) -> dict[str, pd.Series]:
    rewards = {}
    for file_path in sorted(experiment_dir.glob("seed_*/monitor.csv")):
        try:
            df = pd.read_csv(file_path, comment="#")
        except pd.errors.EmptyDataError:
            continue
        if "r" not in df:
            continue
        seed_name = file_path.parent.name
        rewards[seed_name] = df["r"].dropna().astype(float).reset_index(drop=True)
    return rewards


def save_learning_curve(spec: ExperimentSpec, rewards: dict[str, pd.Series], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    for seed_name, series in rewards.items():
        if series.empty:
            continue
        ax.plot(np.arange(1, len(series) + 1), series, linewidth=1.2, alpha=0.8, label=seed_name)
    ax.set_title(f"{spec.label} - TD3 learning curves ({len(rewards)} seed logs)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.grid(True, alpha=0.28)
    if rewards:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = output_dir / f"{spec.folder}_learning_curves.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def scenario_seed_means(metrics: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        metrics.groupby(["seed", "scenario"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )


def save_mean_iae_bar(spec: ExperimentSpec, metrics: pd.DataFrame, output_dir: Path) -> Path:
    per_seed = scenario_seed_means(metrics, "IAE")
    summary = (
        per_seed.groupby("scenario", dropna=False)["IAE"]
        .agg(["mean", "std"])
        .reindex(SCENARIO_ORDER)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["scenario"], summary["mean"], yerr=summary["std"].fillna(0.0), capsize=6)
    ax.set_title(f"{spec.label} - mean IAE with error bars ({metrics['seed'].nunique()} completed seeds)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean IAE")
    ax.grid(True, axis="y", alpha=0.28)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    out = output_dir / f"{spec.folder}_mean_iae_error_bars.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def save_boxplot(spec: ExperimentSpec, metrics: pd.DataFrame, output_dir: Path) -> Path:
    per_seed = scenario_seed_means(metrics, "IAE")
    data = [
        per_seed.loc[per_seed["scenario"] == scenario, "IAE"].to_numpy()
        for scenario in SCENARIO_ORDER
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=SCENARIO_ORDER, showmeans=True)
    ax.set_title(f"{spec.label} - TD3 IAE distribution across seeds")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Per-seed mean IAE")
    ax.grid(True, axis="y", alpha=0.28)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    out = output_dir / f"{spec.folder}_seed_performance_boxplot.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_table_rows(metrics: pd.DataFrame) -> list[list[str]]:
    per_seed = scenario_seed_means(metrics, "IAE")
    rows: list[list[str]] = []
    for scenario in SCENARIO_ORDER:
        vals = per_seed.loc[per_seed["scenario"] == scenario, "IAE"]
        rows.append([scenario, f"{vals.mean():.3f} +/- {vals.std():.3f}"])

    hard = metrics[
        metrics["condition"].isin(HARD_CONDITIONS)
        & metrics["scenario"].isin(HARD_SCENARIOS)
    ]
    for col in ("IAE", "SSE", "ControlEnergy"):
        per_seed_hard = hard.groupby("seed", dropna=False)[col].mean()
        rows.append([f"hard-region {col}", f"{per_seed_hard.mean():.3f} +/- {per_seed_hard.std():.3f}"])
    if "SaturationFraction" in hard:
        per_seed_sat = hard.groupby("seed", dropna=False)["SaturationFraction"].mean()
        rows.append(["hard-region SaturationFraction", f"{per_seed_sat.mean():.3f} +/- {per_seed_sat.std():.3f}"])
    return rows


def save_mean_std_table(spec: ExperimentSpec, metrics: pd.DataFrame, output_dir: Path) -> Path:
    rows = make_table_rows(metrics)
    table_df = pd.DataFrame(rows, columns=["Metric", "Mean +/- Std"])
    table_df.to_csv(output_dir / f"{spec.folder}_mean_std_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 0.48 * len(rows) + 1.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.28)
    ax.set_title(f"{spec.label} - mean +/- standard deviation")
    fig.tight_layout()
    out = output_dir / f"{spec.folder}_mean_std_table.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def hard_region_summary(metrics: pd.DataFrame) -> dict[str, float]:
    hard = metrics[
        metrics["condition"].isin(HARD_CONDITIONS)
        & metrics["scenario"].isin(HARD_SCENARIOS)
    ]
    summary = {
        "HardMeanIAE": hard["IAE"].mean(),
        "HardMeanSSE": hard["SSE"].mean(),
        "HardMeanEnergy": hard["ControlEnergy"].mean(),
    }
    if "SaturationFraction" in hard:
        summary["HardSaturationFraction"] = hard["SaturationFraction"].mean()
    else:
        summary["HardSaturationFraction"] = np.nan
    return summary


def save_cross_experiment_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    columns = ["HardMeanIAE", "HardMeanSSE", "HardMeanEnergy"]
    titles = ["Hard-region IAE", "Hard-region SSE", "Hard-region energy"]
    labels = summary["ExperimentShort"]
    for ax, col, title in zip(axes, columns, titles, strict=True):
        ax.bar(labels, summary[col])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.28)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("All experiments - hard nonlinear region comparison")
    fig.tight_layout()
    out = output_dir / "all_experiments_hard_region_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    outputs_root = root / "outputs"
    graph_root = outputs_root / "four_experiment_graphs"
    graph_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_rows = []
    for idx, spec in enumerate(EXPERIMENTS, start=1):
        experiment_dir = outputs_root / spec.folder
        metrics = read_metrics(experiment_dir)
        rewards = read_rewards(experiment_dir)
        if metrics.empty:
            continue

        experiment_output = graph_root / spec.folder
        experiment_output.mkdir(parents=True, exist_ok=True)

        generated = [
            save_learning_curve(spec, rewards, experiment_output),
            save_mean_iae_bar(spec, metrics, experiment_output),
            save_boxplot(spec, metrics, experiment_output),
            save_mean_std_table(spec, metrics, experiment_output),
        ]
        for path in generated:
            manifest_rows.append(
                {
                    "Experiment": spec.label,
                    "CompletedMetricSeeds": int(metrics["seed"].nunique()),
                    "RewardLogs": len(rewards),
                    "Graph": str(path),
                }
            )
        hard_summary = hard_region_summary(metrics)
        summary_rows.append(
            {
                "Experiment": spec.label,
                "ExperimentShort": f"E{idx}",
                "CompletedMetricSeeds": int(metrics["seed"].nunique()),
                **hard_summary,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(graph_root / "graph_manifest.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(graph_root / "all_experiments_hard_region_summary.csv", index=False)
    save_cross_experiment_summary(summary, graph_root)
    print(f"Generated graphs in {graph_root}")


if __name__ == "__main__":
    main()
