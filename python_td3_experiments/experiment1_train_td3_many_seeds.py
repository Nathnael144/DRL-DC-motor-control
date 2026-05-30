from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent / "outputs" / ".mplcache_bootstrap").resolve()),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from dc_motor_env import (
    DCMotorSpeedEnv,
    MotorParams,
    ResidualLQRDCMotorEnv,
    SCENARIO_NAMES,
    get_condition_library,
)


class EpisodeRewardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode and "r" in episode:
                self.episode_rewards.append(float(episode["r"]))
        return True


class SeedProgressCallback(BaseCallback):
    def __init__(
        self,
        seed: int,
        total_timesteps: int,
        seed_dir: Path,
        print_every_steps: int,
        chunk_index_getter,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.seed_dir = seed_dir
        self.print_every_steps = max(1, print_every_steps)
        self.chunk_index_getter = chunk_index_getter
        self.last_print = 0
        self.seed_start_time = time.time()

    def _on_step(self) -> bool:
        current_steps = int(self.num_timesteps)
        if current_steps - self.last_print >= self.print_every_steps:
            self.last_print = current_steps
            elapsed = time.time() - self.seed_start_time
            pct = 100.0 * current_steps / max(1, self.total_timesteps)
            msg = (
                f"[seed {self.seed:02d}] chunk {self.chunk_index_getter()} "
                f"progress: {current_steps}/{self.total_timesteps} steps "
                f"({pct:.1f}%), elapsed {elapsed/60.0:.1f} min"
            )
            print(msg, flush=True)
            progress_payload = {
                "seed": self.seed,
                "timesteps_done": current_steps,
                "timesteps_total": self.total_timesteps,
                "percent_complete": pct,
                "elapsed_seconds": elapsed,
                "chunk_index": self.chunk_index_getter(),
            }
            (self.seed_dir / "progress.json").write_text(
                json.dumps(progress_payload, indent=2),
                encoding="utf-8",
            )
        return True


def compute_metrics(t: np.ndarray, ref: np.ndarray, speed: np.ndarray, voltage: np.ndarray) -> dict[str, float]:
    error = ref - speed
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    iae = float(np.trapezoid(np.abs(error), t))
    ise = float(np.trapezoid(error**2, t))
    sse = float(abs(error[-1]))
    control_energy = float(np.trapezoid(voltage**2, t))
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    abs_voltage = np.abs(voltage)
    mean_abs_voltage = float(np.trapezoid(abs_voltage, t) / duration) if duration > 0 else 0.0
    max_abs_voltage = float(np.max(abs_voltage)) if len(abs_voltage) else 0.0
    near_saturation_fraction = float(np.mean(abs_voltage >= 0.85 * 24.0)) if len(abs_voltage) else 0.0
    saturation_fraction = float(np.mean(abs_voltage >= 0.95 * 24.0)) if len(abs_voltage) else 0.0

    final_ref = float(ref[-1])
    constant_reference = bool(np.allclose(ref, final_ref))
    rise_time = np.nan
    settling_time = np.nan
    overshoot = np.nan
    if constant_reference and final_ref != 0.0:
        low = 0.1 * final_ref
        high = 0.9 * final_ref
        above_low = np.where(speed >= low)[0]
        above_high = np.where(speed >= high)[0]
        if len(above_low) > 0 and len(above_high) > 0:
            rise_time = float(t[above_high[0]] - t[above_low[0]])

        band = 0.02 * abs(final_ref)
        outside = np.where(np.abs(error) > band)[0]
        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] < len(t) - 1:
            settling_time = float(t[outside[-1] + 1])

        overshoot = float(max(0.0, (np.max(speed) - final_ref) / abs(final_ref) * 100.0))

    return {
        "IAE": iae,
        "ISE": ise,
        "SSE": sse,
        "ControlEnergy": control_energy,
        "MeanAbsVoltage": mean_abs_voltage,
        "MaxAbsVoltage": max_abs_voltage,
        "NearSaturationFraction": near_saturation_fraction,
        "SaturationFraction": saturation_fraction,
        "RiseTime": rise_time,
        "SettlingTime": settling_time,
        "Overshoot": overshoot,
        "dt": dt,
    }


def make_motor_env(
    *,
    control_mode: str,
    residual_voltage_limit: float,
    **kwargs: Any,
) -> DCMotorSpeedEnv:
    if control_mode == "direct":
        return DCMotorSpeedEnv(**kwargs)
    if control_mode == "residual_lqr":
        return ResidualLQRDCMotorEnv(
            residual_voltage_limit=residual_voltage_limit,
            **kwargs,
        )
    raise ValueError(f"Unknown control_mode: {control_mode}")


def evaluate_model(
    model: Any,
    output_dir: Path,
    seed: int,
    control_mode: str,
    residual_voltage_limit: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    conditions = get_condition_library()
    for condition_name, condition in conditions.items():
        for scenario_name in SCENARIO_NAMES:
            env = make_motor_env(
                control_mode=control_mode,
                residual_voltage_limit=residual_voltage_limit,
                condition=condition,
                scenario_name=scenario_name,
                training_mode=False,
            )
            obs, _ = env.reset(seed=seed)

            times = [0.0]
            refs = [env._reference(env.t)]
            speeds = [env.state[1]]
            voltages = [0.0]

            done = False
            truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                times.append(env.t)
                refs.append(info["reference"])
                speeds.append(info["speed"])
                voltages.append(info["voltage"])

            t = np.array(times, dtype=np.float64)
            ref = np.array(refs, dtype=np.float64)
            speed = np.array(speeds, dtype=np.float64)
            voltage = np.array(voltages, dtype=np.float64)
            metrics = compute_metrics(t, ref, speed, voltage)
            row: dict[str, float | int | str] = {
                "seed": seed,
                "condition": condition_name,
                "scenario": scenario_name,
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"seed_{seed:02d}_metrics.csv", index=False)
    return df


def train_one_seed(
    seed: int,
    total_timesteps: int,
    output_root: Path,
    chunk_timesteps: int,
    print_every_steps: int,
    training_condition_mode: str,
    training_scenario_mode: str,
    algorithm: str,
    reward_profile: str,
    control_mode: str,
    residual_voltage_limit: float,
    expert_warmstart_steps: int,
    expert_controller: str,
    curriculum: str,
) -> tuple[pd.DataFrame, list[float]]:
    seed_dir = output_root / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    def make_env() -> DCMotorSpeedEnv:
        monitor_file = seed_dir / "monitor.csv"
        env = make_motor_env(
            control_mode=control_mode,
            residual_voltage_limit=residual_voltage_limit,
            training_mode=True,
            training_condition_mode=training_condition_mode,
            training_scenario_mode=training_scenario_mode,
            reward_profile=reward_profile,
        )
        return Monitor(env, filename=str(monitor_file))

    vec_env = DummyVecEnv([make_env])
    vec_env.seed(seed)

    model = make_model(algorithm=algorithm, vec_env=vec_env, seed=seed)
    model_name = algorithm.upper()
    model_prefix_name = model_prefix(algorithm)

    if expert_warmstart_steps > 0:
        warmstart_replay_buffer(
            model=model,
            seed=seed,
            total_steps=expert_warmstart_steps,
            control_mode=control_mode,
            residual_voltage_limit=residual_voltage_limit,
            training_condition_mode=training_condition_mode,
            training_scenario_mode=training_scenario_mode,
            reward_profile=reward_profile,
            expert_controller=expert_controller,
        )

    print(f"{model_name} agent created for seed {seed:02d}.", flush=True)
    print(f"Training seed {seed:02d}: up to {total_timesteps} timesteps.", flush=True)

    reward_callback = EpisodeRewardCallback()
    timesteps_done = 0
    chunk_index = 0

    while timesteps_done < total_timesteps:
        chunk_index += 1
        current_condition_mode, current_scenario_mode = curriculum_modes(
            curriculum=curriculum,
            base_condition_mode=training_condition_mode,
            base_scenario_mode=training_scenario_mode,
            progress=timesteps_done / max(1, total_timesteps),
        )
        set_training_modes(vec_env, current_condition_mode, current_scenario_mode)
        next_boundary = next_curriculum_boundary(curriculum, timesteps_done, total_timesteps)
        this_chunk = min(
            chunk_timesteps,
            total_timesteps - timesteps_done,
            max(1, next_boundary - timesteps_done),
        )
        print(
            f"--- Seed {seed:02d} chunk {chunk_index}: "
            f"timesteps {timesteps_done + 1}..{timesteps_done + this_chunk}, "
            f"condition_mode={current_condition_mode}, scenario_mode={current_scenario_mode} ---",
            flush=True,
        )
        progress_callback = SeedProgressCallback(
            seed=seed,
            total_timesteps=total_timesteps,
            seed_dir=seed_dir,
            print_every_steps=print_every_steps,
            chunk_index_getter=lambda: chunk_index,
        )
        model.learn(
            total_timesteps=this_chunk,
            progress_bar=False,
            reset_num_timesteps=False,
            callback=[reward_callback, progress_callback],
        )
        timesteps_done += this_chunk
        checkpoint_path = seed_dir / f"{model_prefix_name}_model_chunk_{chunk_index:03d}"
        model.save(checkpoint_path)
        model.save(seed_dir / f"{model_prefix_name}_model_latest")
        chunk_payload = {
            "seed": seed,
            "chunk_index": chunk_index,
            "timesteps_done": timesteps_done,
            "timesteps_total": total_timesteps,
            "condition_mode": current_condition_mode,
            "scenario_mode": current_scenario_mode,
            "episode_rewards_seen": len(reward_callback.episode_rewards),
            "last_episode_reward": reward_callback.episode_rewards[-1] if reward_callback.episode_rewards else None,
        }
        (seed_dir / "chunk_status.json").write_text(
            json.dumps(chunk_payload, indent=2),
            encoding="utf-8",
        )
        print(f"Checkpoint saved after {timesteps_done} timesteps.", flush=True)
        if reward_callback.episode_rewards:
            recent_rewards = reward_callback.episode_rewards[-10:]
            avg_reward = float(np.mean(recent_rewards))
            print(
                f"Seed {seed:02d} chunk {chunk_index} complete. "
                f"Recent avg episode reward: {avg_reward:.3f}",
                flush=True,
            )

    model.save(seed_dir / f"{model_prefix_name}_model")
    metrics_df = evaluate_model(
        model,
        seed_dir,
        seed,
        control_mode=control_mode,
        residual_voltage_limit=residual_voltage_limit,
    )
    return metrics_df, reward_callback.episode_rewards


def model_prefix(algorithm: str) -> str:
    return algorithm.lower()


def make_model(algorithm: str, vec_env: DummyVecEnv, seed: int) -> Any:
    algorithm = algorithm.upper()
    if algorithm == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(1, dtype=np.float32),
            sigma=0.1 * np.ones(1, dtype=np.float32),
        )
        return TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-3,
            buffer_size=100_000,
            learning_starts=2_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 256]},
            seed=seed,
            verbose=0,
        )

    if algorithm == "SAC":
        return SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=250_000,
            learning_starts=2_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=(5, "step"),
            gradient_steps=1,
            ent_coef="auto_0.05",
            target_update_interval=1,
            policy_kwargs={"net_arch": [128, 128]},
            seed=seed,
            verbose=0,
        )

    if algorithm == "DDPG":
        action_noise = NormalActionNoise(
            mean=np.zeros(1, dtype=np.float32),
            sigma=0.15 * np.ones(1, dtype=np.float32),
        )
        return DDPG(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=250_000,
            learning_starts=2_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=(5, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [128, 128]},
            seed=seed,
            verbose=0,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def make_expert_controller(expert_controller: str) -> Any:
    from experiment2_add_controllers import LQRController, MPCLinearClippedController

    if expert_controller == "lqr":
        return LQRController(MotorParams())
    elif expert_controller == "mpc":
        return MPCLinearClippedController(MotorParams())
    raise ValueError(f"Unknown expert_controller: {expert_controller}")


def expert_voltage(env: DCMotorSpeedEnv, controller: Any) -> float:
    return float(controller.act(env))


def expert_action_for_env(
    env: DCMotorSpeedEnv,
    controller: Any,
    control_mode: str,
    residual_voltage_limit: float,
) -> np.ndarray:
    voltage = expert_voltage(env, controller)
    if control_mode == "direct":
        normalized = voltage / env.params.v_max
    elif control_mode == "residual_lqr":
        baseline = float(env._lqr_voltage())  # type: ignore[attr-defined]
        normalized = (voltage - baseline) / max(residual_voltage_limit, 1e-6)
    else:
        raise ValueError(f"Unknown control_mode: {control_mode}")
    return np.array([np.clip(normalized, -1.0, 1.0)], dtype=np.float32)


def warmstart_replay_buffer(
    model: Any,
    seed: int,
    total_steps: int,
    control_mode: str,
    residual_voltage_limit: float,
    training_condition_mode: str,
    training_scenario_mode: str,
    reward_profile: str,
    expert_controller: str,
) -> None:
    env = make_motor_env(
        control_mode=control_mode,
        residual_voltage_limit=residual_voltage_limit,
        training_mode=True,
        training_condition_mode=training_condition_mode,
        training_scenario_mode=training_scenario_mode,
        reward_profile=reward_profile,
    )
    obs, _ = env.reset(seed=100_000 + seed)
    controller = make_expert_controller(expert_controller)
    print(
        f"Warm-starting replay buffer with {total_steps} {expert_controller.upper()} expert steps.",
        flush=True,
    )
    for step in range(total_steps):
        action = expert_action_for_env(
            env=env,
            controller=controller,
            control_mode=control_mode,
            residual_voltage_limit=residual_voltage_limit,
        )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        model.replay_buffer.add(
            obs.reshape((1, -1)),
            next_obs.reshape((1, -1)),
            action.reshape((1, -1)),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [info],
        )
        obs = next_obs
        if done:
            if hasattr(controller, "reset"):
                controller.reset()
            obs, _ = env.reset()
        if (step + 1) % 10_000 == 0 or step + 1 == total_steps:
            print(f"Warm-start progress: {step + 1}/{total_steps}", flush=True)


def curriculum_modes(
    curriculum: str,
    base_condition_mode: str,
    base_scenario_mode: str,
    progress: float,
) -> tuple[str, str]:
    if curriculum == "none":
        return base_condition_mode, base_scenario_mode
    if curriculum == "nominal_to_hard":
        if progress < 0.25:
            return "nominal", "uniform"
        if progress < 0.50:
            return "robust", "uniform"
        if progress < 0.75:
            return "hard", "uniform"
        return "hard", "hard_tracking"
    raise ValueError(f"Unknown curriculum: {curriculum}")


def next_curriculum_boundary(curriculum: str, timesteps_done: int, total_timesteps: int) -> int:
    if curriculum == "none":
        return total_timesteps
    boundaries = [
        int(np.ceil(total_timesteps * value))
        for value in (0.25, 0.50, 0.75, 1.0)
    ]
    for boundary in boundaries:
        if boundary > timesteps_done:
            return boundary
    return total_timesteps


def set_training_modes(vec_env: DummyVecEnv, condition_mode: str, scenario_mode: str) -> None:
    for wrapped_env in vec_env.envs:
        env = getattr(wrapped_env, "env", wrapped_env)
        env.training_condition_mode = condition_mode
        env.training_scenario_mode = scenario_mode


def is_seed_complete(output_root: Path, seed: int, total_timesteps: int, algorithm: str = "TD3") -> bool:
    seed_dir = output_root / f"seed_{seed:02d}"
    status_file = seed_dir / "chunk_status.json"
    metrics_file = seed_dir / f"seed_{seed:02d}_metrics.csv"
    model_file = seed_dir / f"{model_prefix(algorithm)}_model.zip"
    if not (status_file.exists() and metrics_file.exists() and model_file.exists()):
        return False
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(status.get("timesteps_done", 0)) >= total_timesteps


def load_seed_metrics(output_root: Path, seed: int) -> pd.DataFrame:
    metrics_file = output_root / f"seed_{seed:02d}" / f"seed_{seed:02d}_metrics.csv"
    return pd.read_csv(metrics_file)


def load_seed_rewards(output_root: Path, seed: int) -> list[float]:
    monitor_file = output_root / f"seed_{seed:02d}" / "monitor.csv"
    if not monitor_file.exists():
        return []
    try:
        monitor_df = pd.read_csv(monitor_file, comment="#")
    except pd.errors.EmptyDataError:
        return []
    if "r" not in monitor_df:
        return []
    return [float(value) for value in monitor_df["r"].dropna().to_list()]


def build_summary(all_metrics: pd.DataFrame, output_root: Path) -> None:
    all_metrics.to_csv(output_root / "all_seed_metrics_long.csv", index=False)

    summary = (
        all_metrics.groupby(["condition", "scenario"], dropna=False)[
            [
                "IAE",
                "ISE",
                "SSE",
                "ControlEnergy",
                "MeanAbsVoltage",
                "MaxAbsVoltage",
                "NearSaturationFraction",
                "SaturationFraction",
                "RiseTime",
                "SettlingTime",
                "Overshoot",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(output_root / "summary_mean_std.csv", index=False)

    scenario_summary = (
        all_metrics.groupby(["scenario"], dropna=False)[
            ["IAE", "ISE", "ControlEnergy", "MeanAbsVoltage", "SaturationFraction"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    scenario_summary.to_csv(output_root / "scenario_summary.csv", index=False)


def plot_learning_curves(training_rewards: list[list[float]], output_root: Path, algorithm: str) -> None:
    plt.figure(figsize=(10, 6))
    for seed, rewards in enumerate(training_rewards):
        if rewards:
            plt.plot(np.arange(1, len(rewards) + 1), rewards, linewidth=1.2, alpha=0.85, label=f"Seed {seed}")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"{algorithm.upper()} Learning Curves Across Seeds")
    if any(training_rewards):
        plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / f"{model_prefix(algorithm)}_learning_curves.png", dpi=200)
    plt.close()


def plot_iae_bar_with_error_bars(all_metrics: pd.DataFrame, output_root: Path) -> None:
    ramp_df = all_metrics[all_metrics["scenario"] == "ramp"].copy()
    summary = (
        ramp_df.groupby("condition", dropna=False)["IAE"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )
    plt.figure(figsize=(10, 6))
    plt.bar(summary["condition"], summary["mean"], yerr=summary["std"].fillna(0.0), capsize=6)
    plt.xlabel("Condition")
    plt.ylabel("Mean IAE")
    plt.title("Mean Ramp IAE Across Seeds with Error Bars")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "mean_ramp_iae_error_bars.png", dpi=200)
    plt.close()


def plot_iae_boxplot(all_metrics: pd.DataFrame, output_root: Path, algorithm: str) -> None:
    ramp_df = all_metrics[all_metrics["scenario"] == "ramp"].copy()
    condition_order = list(ramp_df.groupby("condition")["IAE"].mean().sort_values().index)
    data = [ramp_df.loc[ramp_df["condition"] == condition, "IAE"].to_numpy() for condition in condition_order]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=condition_order)
    plt.xlabel("Condition")
    plt.ylabel("IAE")
    plt.title(f"{algorithm.upper()} Ramp IAE Distribution Across Seeds")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / f"{model_prefix(algorithm)}_ramp_iae_boxplot.png", dpi=200)
    plt.close()


def plot_mean_std_table(all_metrics: pd.DataFrame, output_root: Path) -> None:
    table_df = (
        all_metrics.groupby("seed", dropna=False)[
            ["IAE", "ISE", "SSE", "ControlEnergy", "MeanAbsVoltage", "SaturationFraction"]
        ]
        .mean()
        .reset_index()
    )
    display_df = table_df.copy()
    for col in ["IAE", "ISE", "SSE", "ControlEnergy", "MeanAbsVoltage", "SaturationFraction"]:
        display_df[col] = display_df[col].map(lambda value: f"{value:.4f}")

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(display_df) + 1.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    plt.title("Per-Seed Mean Metrics")
    plt.tight_layout()
    plt.savefig(output_root / "seed_mean_metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close()


def generate_graphs(
    all_metrics: pd.DataFrame,
    training_rewards: list[list[float]],
    output_root: Path,
    algorithm: str,
) -> None:
    plot_learning_curves(training_rewards, output_root, algorithm)
    plot_iae_bar_with_error_bars(all_metrics, output_root)
    plot_iae_boxplot(all_metrics, output_root, algorithm)
    plot_mean_std_table(all_metrics, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed deep RL training on DC motor control.")
    parser.add_argument(
        "--algorithm",
        choices=("TD3", "SAC", "DDPG"),
        default="TD3",
        help="Off-policy continuous-control algorithm to train.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds to train.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps per seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python_td3_experiments") / "outputs" / "experiment1_many_seeds",
        help="Output folder for models and CSV metrics.",
    )
    parser.add_argument(
        "--chunk-timesteps",
        type=int,
        default=100_000,
        help="Checkpoint and print progress after each chunk of timesteps.",
    )
    parser.add_argument(
        "--print-every-steps",
        type=int,
        default=10_000,
        help="Print live progress every N environment steps within a seed.",
    )
    parser.add_argument(
        "--training-condition-mode",
        choices=("nominal", "robust", "hard"),
        default="nominal",
        help="Plant-condition distribution used during training.",
    )
    parser.add_argument(
        "--training-scenario-mode",
        choices=("uniform", "hard_tracking"),
        default="uniform",
        help="Reference-scenario distribution used during training.",
    )
    parser.add_argument(
        "--reward-profile",
        choices=("balanced", "tracking_first", "competitive"),
        default="balanced",
        help="Training reward profile. Use tracking_first when testing whether RL can beat classical tracking.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("direct", "residual_lqr"),
        default="direct",
        help="Use direct voltage control or learn a residual correction on top of nominal LQR.",
    )
    parser.add_argument(
        "--residual-voltage-limit",
        type=float,
        default=8.0,
        help="Maximum absolute RL correction voltage when --control-mode residual_lqr is used.",
    )
    parser.add_argument(
        "--expert-warmstart-steps",
        type=int,
        default=0,
        help="Number of LQR/MPC expert transitions to add to the replay buffer before RL training.",
    )
    parser.add_argument(
        "--expert-controller",
        choices=("lqr", "mpc"),
        default="lqr",
        help="Expert controller used for replay-buffer warm-start.",
    )
    parser.add_argument(
        "--curriculum",
        choices=("none", "nominal_to_hard"),
        default="none",
        help="Training curriculum. nominal_to_hard progresses from nominal to hard nonlinear tracking.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First seed index to consider. Use 1 to continue after seed_00.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed seed folders instead of retraining them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache_dir = args.output_dir / ".mplcache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    metadata = {
        "algorithm": args.algorithm,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "motor_params": asdict(MotorParams()),
        "conditions": {k: asdict(v) for k, v in get_condition_library().items()},
        "scenarios": list(SCENARIO_NAMES),
        "training_condition_mode": args.training_condition_mode,
        "training_scenario_mode": args.training_scenario_mode,
        "reward_profile": args.reward_profile,
        "control_mode": args.control_mode,
        "residual_voltage_limit": args.residual_voltage_limit,
        "expert_warmstart_steps": args.expert_warmstart_steps,
        "expert_controller": args.expert_controller,
        "curriculum": args.curriculum,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_frames: list[pd.DataFrame] = []
    training_rewards: list[list[float]] = []

    for seed in range(0, min(args.start_seed, args.seeds)):
        if is_seed_complete(args.output_dir, seed, args.timesteps, args.algorithm):
            print(f"=== Reusing completed seed {seed} / {args.seeds - 1} ===", flush=True)
            all_frames.append(load_seed_metrics(args.output_dir, seed))
            training_rewards.append(load_seed_rewards(args.output_dir, seed))

    for seed in range(args.start_seed, args.seeds):
        if args.skip_completed and is_seed_complete(args.output_dir, seed, args.timesteps, args.algorithm):
            print(f"=== Reusing completed seed {seed} / {args.seeds - 1} ===", flush=True)
            all_frames.append(load_seed_metrics(args.output_dir, seed))
            training_rewards.append(load_seed_rewards(args.output_dir, seed))
            continue

        print(f"=== Training seed {seed} / {args.seeds - 1} ===", flush=True)
        df, rewards = train_one_seed(
            seed=seed,
            total_timesteps=args.timesteps,
            output_root=args.output_dir,
            chunk_timesteps=args.chunk_timesteps,
            print_every_steps=args.print_every_steps,
            training_condition_mode=args.training_condition_mode,
            training_scenario_mode=args.training_scenario_mode,
            algorithm=args.algorithm,
            reward_profile=args.reward_profile,
            control_mode=args.control_mode,
            residual_voltage_limit=args.residual_voltage_limit,
            expert_warmstart_steps=args.expert_warmstart_steps,
            expert_controller=args.expert_controller,
            curriculum=args.curriculum,
        )
        all_frames.append(df)
        training_rewards.append(rewards)

    if not all_frames:
        raise RuntimeError("No seed metrics were available or generated.")

    all_metrics = pd.concat(all_frames, ignore_index=True)
    build_summary(all_metrics, args.output_dir)
    generate_graphs(all_metrics, training_rewards, args.output_dir, args.algorithm)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
