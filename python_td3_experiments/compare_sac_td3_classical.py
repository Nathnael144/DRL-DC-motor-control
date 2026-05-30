"""
Compare balanced SAC against TD3 and classical controllers.

This script expects:
  - SAC multi-seed output from experiment1_train_td3_many_seeds.py
  - TD3 multi-seed output from experiment1_train_td3_many_seeds.py
  - Classical-controller output from experiment2_add_controllers.py

It writes mean +/- std tables and comparison plots suitable for the paper.
For SAC/TD3, the primary std is across seed-level means.  For deterministic
classical controllers, the seed std is left blank and case std is reported.
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


METRIC_COLUMNS = [
    "IAE",
    "ISE",
    "SSE",
    "ControlEnergy",
    "MeanAbsVoltage",
    "SaturationFraction",
]


CLASSICAL_NAME_MAP = {
    "PID": "PID",
    "LQR": "LQR",
    "LQI": "LQI",
    "MPC_linear_clipped": "MPC",
}


def load_deep_rl_metrics(output_dir: Path, controller_name: str) -> pd.DataFrame:
    long_file = output_dir / "all_seed_metrics_long.csv"
    if long_file.exists():
        df = pd.read_csv(long_file)
    else:
        frames = []
        for metrics_file in sorted(output_dir.glob("seed_*/seed_*_metrics.csv")):
            frames.append(pd.read_csv(metrics_file))
        if not frames:
            raise FileNotFoundError(f"No seed metrics found in {output_dir}")
        df = pd.concat(frames, ignore_index=True)

    df = df.copy()
    df["controller"] = controller_name
    df["family"] = "RL"
    return df[["controller", "family", "seed", "condition", "scenario", *METRIC_COLUMNS]]


def load_classical_metrics(classical_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(classical_csv)
    raw = raw[raw["controller"].isin(CLASSICAL_NAME_MAP.keys())].copy()
    raw["controller"] = raw["controller"].map(CLASSICAL_NAME_MAP)
    raw["family"] = "Classical"
    raw["seed"] = np.nan
    rename = {
        "iae": "IAE",
        "ise": "ISE",
        "steady_state_error": "SSE",
        "control_energy": "ControlEnergy",
        "saturation_fraction": "SaturationFraction",
    }
    raw = raw.rename(columns=rename)
    if "MeanAbsVoltage" not in raw:
        raw["MeanAbsVoltage"] = np.nan
    return raw[["controller", "family", "seed", "condition", "scenario", *METRIC_COLUMNS]]


def build_overall_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rl = all_metrics[all_metrics["family"] == "RL"].copy()
    classical = all_metrics[all_metrics["family"] == "Classical"].copy()

    seed_means = (
        rl.groupby(["controller", "seed"], dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    rl_summary = (
        seed_means.groupby("controller", dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    rl_summary.columns = [
        "controller" if col[0] == "controller" else f"{col[0]}_seed_{col[1]}"
        for col in rl_summary.columns
    ]
    rl_summary["family"] = "RL"
    rl_summary["n_seeds"] = seed_means.groupby("controller")["seed"].nunique().reindex(rl_summary["controller"]).to_numpy()

    classical_summary = (
        classical.groupby("controller", dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    classical_summary.columns = [
        "controller" if col[0] == "controller" else f"{col[0]}_case_{col[1]}"
        for col in classical_summary.columns
    ]
    classical_summary["family"] = "Classical"
    classical_summary["n_seeds"] = 0

    rows = []
    for _, row in rl_summary.iterrows():
        payload = {
            "controller": row["controller"],
            "family": row["family"],
            "n_seeds": int(row["n_seeds"]),
        }
        for metric in METRIC_COLUMNS:
            payload[f"{metric}_mean"] = row[f"{metric}_seed_mean"]
            payload[f"{metric}_std_across_seeds"] = row[f"{metric}_seed_std"]
            payload[f"{metric}_std_across_cases"] = np.nan
        rows.append(payload)

    for _, row in classical_summary.iterrows():
        payload = {
            "controller": row["controller"],
            "family": row["family"],
            "n_seeds": 0,
        }
        for metric in METRIC_COLUMNS:
            payload[f"{metric}_mean"] = row[f"{metric}_case_mean"]
            payload[f"{metric}_std_across_seeds"] = np.nan
            payload[f"{metric}_std_across_cases"] = row[f"{metric}_case_std"]
        rows.append(payload)

    summary = pd.DataFrame(rows)
    return summary.sort_values("IAE_mean").reset_index(drop=True)


def build_case_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        all_metrics.groupby(["controller", "family", "condition", "scenario"], dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )


def build_winner_table(case_summary: pd.DataFrame) -> pd.DataFrame:
    flat = case_summary.copy()
    flat.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in flat.columns
    ]
    rows = []
    for (condition, scenario), group in flat.groupby(["condition", "scenario"], dropna=False):
        iae_winner = group.loc[group["IAE_mean"].idxmin()]
        energy_winner = group.loc[group["ControlEnergy_mean"].idxmin()]
        sse_winner = group.loc[group["SSE_mean"].idxmin()]
        rows.append(
            {
                "condition": condition,
                "scenario": scenario,
                "best_iae_controller": iae_winner["controller"],
                "best_iae": iae_winner["IAE_mean"],
                "best_energy_controller": energy_winner["controller"],
                "best_control_energy": energy_winner["ControlEnergy_mean"],
                "best_sse_controller": sse_winner["controller"],
                "best_sse": sse_winner["SSE_mean"],
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "scenario"]).reset_index(drop=True)


def build_hard_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    hard = all_metrics[all_metrics["condition"].isin(["saturation", "combined_stress"])].copy()
    hard_seed = (
        hard[hard["family"] == "RL"]
        .groupby(["controller", "seed"], dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    hard_rl = (
        hard_seed.groupby("controller", dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    hard_rl.columns = [
        "controller" if col[0] == "controller" else f"{col[0]}_seed_{col[1]}"
        for col in hard_rl.columns
    ]
    hard_rl["family"] = "RL"

    hard_classical = (
        hard[hard["family"] == "Classical"]
        .groupby("controller", dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    hard_classical.columns = [
        "controller" if col[0] == "controller" else f"{col[0]}_case_{col[1]}"
        for col in hard_classical.columns
    ]
    hard_classical["family"] = "Classical"

    rows = []
    for _, row in hard_rl.iterrows():
        payload = {"controller": row["controller"], "family": "RL"}
        for metric in METRIC_COLUMNS:
            payload[f"{metric}_mean"] = row[f"{metric}_seed_mean"]
            payload[f"{metric}_std_across_seeds"] = row[f"{metric}_seed_std"]
        rows.append(payload)

    for _, row in hard_classical.iterrows():
        payload = {"controller": row["controller"], "family": "Classical"}
        for metric in METRIC_COLUMNS:
            payload[f"{metric}_mean"] = row[f"{metric}_case_mean"]
            payload[f"{metric}_std_across_seeds"] = np.nan
        rows.append(payload)

    return pd.DataFrame(rows).sort_values("IAE_mean").reset_index(drop=True)


def plot_bar(summary: pd.DataFrame, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    ordered = summary.sort_values(f"{metric}_mean")
    colors = ["#2E86AB" if family == "RL" else "#7A7A7A" for family in ordered["family"]]
    yerr = ordered[f"{metric}_std_across_seeds"].fillna(0.0).to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    ax.bar(ordered["controller"], ordered[f"{metric}_mean"], yerr=yerr, capsize=5, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_overall(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_bar(
        summary,
        "IAE",
        output_dir / "overall_mean_iae_with_seed_std.png",
        "Overall Mean IAE: SAC vs TD3 vs Classical",
        "Mean IAE",
    )
    plot_bar(
        summary,
        "ControlEnergy",
        output_dir / "overall_mean_energy_with_seed_std.png",
        "Overall Mean Control Energy",
        "Mean control energy",
    )
    plot_bar(
        summary,
        "SSE",
        output_dir / "overall_mean_sse_with_seed_std.png",
        "Overall Mean Steady-State Error",
        "Mean SSE",
    )


def plot_hard(hard_summary: pd.DataFrame, output_dir: Path) -> None:
    plot_bar(
        hard_summary,
        "IAE",
        output_dir / "hard_cases_mean_iae_with_seed_std.png",
        "Hard Nonlinear Cases Mean IAE",
        "Mean IAE on saturation + combined stress",
    )
    plot_bar(
        hard_summary,
        "ControlEnergy",
        output_dir / "hard_cases_mean_energy_with_seed_std.png",
        "Hard Nonlinear Cases Mean Control Energy",
        "Mean control energy on hard cases",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sac-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "sac_balanced_10seeds_100k",
    )
    parser.add_argument(
        "--td3-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "experiment4_condition_aware_10seeds_100k",
    )
    parser.add_argument(
        "--classical-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "experiment2_more_controllers_full_rl" / "controller_metrics_long.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "sac_vs_td3_classical_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sac = load_deep_rl_metrics(args.sac_dir, "SAC")
    td3 = load_deep_rl_metrics(args.td3_dir, "TD3")
    classical = load_classical_metrics(args.classical_csv)
    all_metrics = pd.concat([sac, td3, classical], ignore_index=True)

    all_metrics.to_csv(args.output_dir / "combined_metrics_long.csv", index=False)
    overall = build_overall_summary(all_metrics)
    case_summary = build_case_summary(all_metrics)
    winners = build_winner_table(case_summary)
    hard = build_hard_summary(all_metrics)

    overall.to_csv(args.output_dir / "overall_mean_std_summary.csv", index=False)
    case_summary.to_csv(args.output_dir / "case_mean_std_summary.csv", index=False)
    winners.to_csv(args.output_dir / "winner_table_by_case.csv", index=False)
    hard.to_csv(args.output_dir / "hard_cases_mean_std_summary.csv", index=False)

    plot_overall(overall, args.output_dir)
    plot_hard(hard, args.output_dir)

    print("[done] wrote comparison outputs to:", args.output_dir, flush=True)
    print("[overall]", flush=True)
    print(
        overall[
            [
                "controller",
                "family",
                "n_seeds",
                "IAE_mean",
                "IAE_std_across_seeds",
                "ControlEnergy_mean",
                "ControlEnergy_std_across_seeds",
                "SSE_mean",
                "SSE_std_across_seeds",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("[winner counts: IAE]", flush=True)
    print(winners["best_iae_controller"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
