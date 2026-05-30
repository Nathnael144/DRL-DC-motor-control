"""
Evaluate saved RL checkpoints and select the best validation checkpoint.

The selector evaluates each chunk checkpoint on all benchmark conditions and
scenarios, then ranks checkpoints by:

    objective = mean_IAE + sse_weight * mean_SSE + energy_weight * mean_ControlEnergy

By default, IAE dominates because tracking is the primary paper metric.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = SCRIPT_DIR / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import pandas as pd
from stable_baselines3 import DDPG, SAC, TD3

from dc_motor_env import SCENARIO_NAMES, get_condition_library
from experiment1_train_td3_many_seeds import compute_metrics, make_motor_env


ALGORITHMS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


def load_model(algorithm: str, checkpoint: Path) -> Any:
    return ALGORITHMS[algorithm.upper()].load(str(checkpoint))


def checkpoint_label(path: Path) -> str:
    if "_chunk_" in path.stem:
        return path.stem.split("_chunk_")[-1]
    if path.stem.endswith("_model"):
        return "final"
    return path.stem


def find_checkpoints(seed_dir: Path, algorithm: str) -> list[Path]:
    prefix = algorithm.lower()
    checkpoints = sorted(seed_dir.glob(f"{prefix}_model_chunk_*.zip"))
    final_model = seed_dir / f"{prefix}_model.zip"
    if final_model.exists():
        checkpoints.append(final_model)
    if not checkpoints:
        raise FileNotFoundError(f"No {algorithm} checkpoints found in {seed_dir}")
    return checkpoints


def evaluate_checkpoint(
    model: Any,
    checkpoint: Path,
    seed_index: int,
    control_mode: str,
    residual_voltage_limit: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for condition_name, condition in get_condition_library().items():
        for scenario_name in SCENARIO_NAMES:
            env = make_motor_env(
                control_mode=control_mode,
                residual_voltage_limit=residual_voltage_limit,
                condition=condition,
                scenario_name=scenario_name,
                training_mode=False,
            )
            obs, _ = env.reset(seed=seed_index)

            times = [0.0]
            refs = [env._reference(env.t)]
            speeds = [env.state[1]]
            voltages = [0.0]

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                times.append(env.t)
                refs.append(info["reference"])
                speeds.append(info["speed"])
                voltages.append(info["voltage"])

            metrics = compute_metrics(
                pd.Series(times).to_numpy(),
                pd.Series(refs).to_numpy(),
                pd.Series(speeds).to_numpy(),
                pd.Series(voltages).to_numpy(),
            )
            row: dict[str, float | int | str] = {
                "seed": seed_index,
                "checkpoint": checkpoint_label(checkpoint),
                "checkpoint_path": str(checkpoint),
                "condition": condition_name,
                "scenario": scenario_name,
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_seed(
    seed_dir: Path,
    algorithm: str,
    seed_index: int,
    control_mode: str,
    residual_voltage_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for checkpoint in find_checkpoints(seed_dir, algorithm):
        print(f"[eval] seed={seed_index:02d} checkpoint={checkpoint.name}", flush=True)
        model = load_model(algorithm, checkpoint)
        frames.append(
            evaluate_checkpoint(
                model=model,
                checkpoint=checkpoint,
                seed_index=seed_index,
                control_mode=control_mode,
                residual_voltage_limit=residual_voltage_limit,
            )
        )
    long_df = pd.concat(frames, ignore_index=True)
    summary = (
        long_df.groupby(["seed", "checkpoint", "checkpoint_path"], dropna=False)[
            ["IAE", "ISE", "SSE", "ControlEnergy", "SaturationFraction"]
        ]
        .mean()
        .reset_index()
    )
    return long_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("SAC", "TD3", "DDPG"), default="SAC")
    parser.add_argument("--control-mode", choices=("direct", "residual_lqr"), default="direct")
    parser.add_argument("--residual-voltage-limit", type=float, default=8.0)
    parser.add_argument("--sse-weight", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_long = []
    all_summary = []
    for seed_dir in sorted(args.run_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed_index = int(seed_dir.name.split("_")[-1])
        long_df, summary = evaluate_seed(
            seed_dir=seed_dir,
            algorithm=args.algorithm,
            seed_index=seed_index,
            control_mode=args.control_mode,
            residual_voltage_limit=args.residual_voltage_limit,
        )
        all_long.append(long_df)
        all_summary.append(summary)

    if not all_summary:
        raise RuntimeError(f"No seed directories found in {args.run_dir}")

    long = pd.concat(all_long, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    summary["objective"] = (
        summary["IAE"]
        + args.sse_weight * summary["SSE"]
        + args.energy_weight * summary["ControlEnergy"]
    )
    best_by_seed = (
        summary.sort_values("objective")
        .groupby("seed", dropna=False)
        .head(1)
        .sort_values("seed")
        .reset_index(drop=True)
    )

    long.to_csv(args.run_dir / "checkpoint_validation_metrics_long.csv", index=False)
    summary.to_csv(args.run_dir / "checkpoint_validation_summary.csv", index=False)
    best_by_seed.to_csv(args.run_dir / "best_checkpoint_by_seed.csv", index=False)
    best_keys = best_by_seed[["seed", "checkpoint"]].copy()
    best_selected = long.merge(best_keys, on=["seed", "checkpoint"], how="inner")
    best_selected_dir = args.run_dir / "best_selected"
    best_selected_dir.mkdir(parents=True, exist_ok=True)
    best_selected.to_csv(best_selected_dir / "all_seed_metrics_long.csv", index=False)

    best_payload = {
        "algorithm": args.algorithm,
        "control_mode": args.control_mode,
        "residual_voltage_limit": args.residual_voltage_limit,
        "sse_weight": args.sse_weight,
        "energy_weight": args.energy_weight,
        "best_checkpoints": best_by_seed.to_dict(orient="records"),
    }
    (args.run_dir / "best_checkpoint_selection.json").write_text(
        json.dumps(best_payload, indent=2),
        encoding="utf-8",
    )

    for _, row in best_by_seed.iterrows():
        src = Path(row["checkpoint_path"])
        dst = args.run_dir / f"seed_{int(row['seed']):02d}" / f"{args.algorithm.lower()}_best_model.zip"
        shutil.copy2(src, dst)

    print("[done] best checkpoints:", flush=True)
    print(best_by_seed.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
