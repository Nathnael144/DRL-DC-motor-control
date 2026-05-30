"""
Experiment 2: add more controllers for comparison.

This script evaluates fixed-design controllers on the same DC-motor
conditions and reference signals used by the TD3 experiments.  Classical
controllers are designed only on the nominal motor model and then reused
without retuning on every test condition.

Controllers included by default:
  - PID
  - LQR
  - LQI
  - MPC_linear_clipped

Optional controllers:
  - TD3 from an existing model path
  - DDPG and SAC trained with Stable-Baselines3

Outputs:
  - controller_metrics_long.csv
  - controller_summary.csv
  - controller_winner_table.csv
  - controller_comparison_heatmap.png
  - mean_iae_by_controller.png
  - mean_control_energy_by_controller.png
  - controller_winner_table.png
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = SCRIPT_DIR / ".matplotlib_cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dc_motor_env import (
    DCMotorSpeedEnv,
    MotorParams,
    SCENARIO_NAMES,
    get_condition_library,
)


VOLTAGE_LIMIT = 24.0


@dataclass
class RolloutMetrics:
    iae: float
    ise: float
    steady_state_error: float
    control_energy: float
    rise_time: float
    settling_time: float
    overshoot_pct: float
    saturation_fraction: float


def motor_state_space(params: MotorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous-time nominal state model x=[current, omega]."""
    a = np.array(
        [
            [-params.ra / params.la, -params.ke / params.la],
            [params.kt / params.j, -params.bm / params.j],
        ],
        dtype=float,
    )
    b = np.array([[1.0 / params.la], [0.0]], dtype=float)
    c = np.array([[0.0, 1.0]], dtype=float)
    return a, b, c


def clip_voltage(voltage: float) -> float:
    return float(np.clip(voltage, -VOLTAGE_LIMIT, VOLTAGE_LIMIT))


def as_env_action(voltage: float) -> np.ndarray:
    return np.array([clip_voltage(voltage) / VOLTAGE_LIMIT], dtype=np.float32)


def care_gain(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a small continuous CARE using the Hamiltonian eigen method."""
    r_inv = np.linalg.inv(r)
    h = np.block(
        [
            [a, -b @ r_inv @ b.T],
            [-q, -a.T],
        ]
    )
    eigvals, eigvecs = np.linalg.eig(h)
    stable = np.where(np.real(eigvals) < 0.0)[0]
    if stable.size != a.shape[0]:
        stable = np.argsort(np.real(eigvals))[: a.shape[0]]

    v = eigvecs[:, stable]
    v1 = v[: a.shape[0], :]
    v2 = v[a.shape[0] :, :]
    p = np.real(v2 @ np.linalg.inv(v1))
    p = 0.5 * (p + p.T)
    k = r_inv @ b.T @ p
    return np.real(k), p


def rk4_linear_discretization(
    a: np.ndarray,
    b: np.ndarray,
    dt: float,
    substeps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the fast electrical model over one controller interval."""
    n = a.shape[0]
    h = dt / substeps

    def step(x: np.ndarray, u: float) -> np.ndarray:
        def f(z: np.ndarray) -> np.ndarray:
            return a @ z + (b[:, 0] * u)

        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    ad = np.zeros((n, n), dtype=float)
    for i in range(n):
        x0 = np.zeros(n, dtype=float)
        x0[i] = 1.0
        x = x0
        for _ in range(substeps):
            x = step(x, 0.0)
        ad[:, i] = x

    x = np.zeros(n, dtype=float)
    for _ in range(substeps):
        x = step(x, 1.0)
    bd = x.reshape(n, 1)
    return ad, bd


def dlqr_gain(
    ad: np.ndarray,
    bd: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    iterations: int = 1000,
    tol: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    p = q.copy()
    for _ in range(iterations):
        gain = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
        p_next = q + ad.T @ p @ (ad - bd @ gain)
        if np.linalg.norm(p_next - p, ord="fro") < tol:
            p = p_next
            break
        p = p_next
    gain = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    return gain, p


def steady_state_for_reference(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    reference: float,
) -> tuple[np.ndarray, float]:
    mat = np.block([[a, b], [c, np.zeros((1, 1))]])
    rhs = np.array([0.0, 0.0, reference], dtype=float)
    sol = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    return sol[:2], float(sol[2])


class BaseController:
    name = "controller"

    def reset(self) -> None:
        pass

    def act(self, env: DCMotorSpeedEnv) -> float:
        raise NotImplementedError


class PIDController(BaseController):
    name = "PID"

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def act(self, env: DCMotorSpeedEnv) -> float:
        _, omega = env.state
        reference = env._reference(env.t)
        error = reference - omega
        dt = env.control_dt
        self.integral = float(np.clip(self.integral + error * dt, -250.0, 250.0))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class LQRController(BaseController):
    name = "LQR"

    def __init__(self, params: MotorParams):
        self.a, self.b, self.c = motor_state_space(params)
        q = np.diag([0.003, 2.8])
        r = np.array([[0.045]])
        self.k, _ = care_gain(self.a, self.b, q, r)
        denom = self.c @ np.linalg.inv(self.a - self.b @ self.k) @ self.b
        self.nbar = float(-1.0 / denom.item())

    def act(self, env: DCMotorSpeedEnv) -> float:
        x = np.asarray(env.state, dtype=float)
        reference = env._reference(env.t)
        return float((-self.k @ x.reshape(-1, 1)).item() + self.nbar * reference)


class LQIController(BaseController):
    name = "LQI"

    def __init__(self, params: MotorParams):
        a, b, c = motor_state_space(params)
        self.c = c
        self.integral = 0.0
        a_aug = np.block(
            [
                [a, np.zeros((2, 1))],
                [-c, np.zeros((1, 1))],
            ]
        )
        b_aug = np.vstack([b, [[0.0]]])
        q_aug = np.diag([0.003, 2.8, 45.0])
        r = np.array([[0.055]])
        self.k, _ = care_gain(a_aug, b_aug, q_aug, r)

    def reset(self) -> None:
        self.integral = 0.0

    def act(self, env: DCMotorSpeedEnv) -> float:
        x = np.asarray(env.state, dtype=float)
        reference = env._reference(env.t)
        error = reference - float((self.c @ x.reshape(-1, 1)).item())
        self.integral = float(np.clip(self.integral + error * env.control_dt, -250.0, 250.0))
        x_aug = np.array([x[0], x[1], self.integral], dtype=float)
        return float((-self.k @ x_aug.reshape(-1, 1)).item())


class MPCLinearClippedController(BaseController):
    name = "MPC_linear_clipped"

    def __init__(self, params: MotorParams, horizon: int = 35):
        self.a, self.b, self.c = motor_state_space(params)
        ad, bd = rk4_linear_discretization(self.a, self.b, dt=0.001, substeps=10)
        q = np.diag([0.004, 4.5])
        r = np.array([[0.075]])
        terminal_gain, terminal_p = dlqr_gain(ad, bd, q, r)
        p = terminal_p
        gains: list[np.ndarray] = []
        for _ in range(horizon):
            gain = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
            gains.append(gain)
            p = q + ad.T @ p @ (ad - bd @ gain)
        gains.reverse()
        self.k0 = gains[0] if gains else terminal_gain

    def act(self, env: DCMotorSpeedEnv) -> float:
        x = np.asarray(env.state, dtype=float)
        reference = env._reference(env.t)
        x_ss, u_ss = steady_state_for_reference(self.a, self.b, self.c, reference)
        return float(u_ss - (self.k0 @ (x - x_ss).reshape(-1, 1)).item())


class SB3Controller(BaseController):
    def __init__(self, name: str, model):
        self.name = name
        self.model = model

    def act(self, env: DCMotorSpeedEnv) -> float:
        obs = env._build_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        return float(np.asarray(action).reshape(-1)[0] * VOLTAGE_LIMIT)


def rollout_controller(
    controller: BaseController,
    condition_name: str,
    scenario_name: str,
    seed: int = 0,
) -> dict:
    condition = get_condition_library()[condition_name]
    env = DCMotorSpeedEnv(
        condition=condition,
        scenario_name=scenario_name,
        training_mode=False,
    )
    _, _ = env.reset(seed=seed)
    controller.reset()

    times: list[float] = []
    references: list[float] = []
    omegas: list[float] = []
    voltages: list[float] = []
    errors: list[float] = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        voltage = clip_voltage(controller.act(env))
        _, _, terminated, truncated, info = env.step(as_env_action(voltage))
        times.append(float(env.t))
        reference = float(info["reference"])
        speed = float(info["speed"])
        references.append(reference)
        omegas.append(speed)
        voltages.append(float(info["voltage"]))
        errors.append(reference - speed)

    metrics = compute_rollout_metrics(
        np.asarray(times),
        np.asarray(references),
        np.asarray(omegas),
        np.asarray(voltages),
        np.asarray(errors),
        env.control_dt,
    )
    return {
        "controller": controller.name,
        "condition": condition_name,
        "scenario": scenario_name,
        "iae": metrics.iae,
        "ise": metrics.ise,
        "steady_state_error": metrics.steady_state_error,
        "control_energy": metrics.control_energy,
        "rise_time": metrics.rise_time,
        "settling_time": metrics.settling_time,
        "overshoot_pct": metrics.overshoot_pct,
        "saturation_fraction": metrics.saturation_fraction,
    }


def compute_rollout_metrics(
    times: np.ndarray,
    references: np.ndarray,
    omegas: np.ndarray,
    voltages: np.ndarray,
    errors: np.ndarray,
    dt: float,
) -> RolloutMetrics:
    abs_error = np.abs(errors)
    iae = float(np.sum(abs_error) * dt)
    ise = float(np.sum(errors**2) * dt)
    control_energy = float(np.sum(voltages**2) * dt)
    steady_state_error = float(np.mean(abs_error[-max(1, int(0.1 * len(abs_error))) :]))
    saturation_fraction = float(np.mean(np.abs(voltages) >= 0.98 * VOLTAGE_LIMIT))

    final_ref = float(np.max(np.abs(references)))
    if final_ref < 1e-9:
        rise_time = math.nan
        overshoot = math.nan
    else:
        target = max(1e-9, 0.9 * final_ref)
        reached = np.where(np.abs(omegas) >= target)[0]
        rise_time = float(times[reached[0]]) if reached.size else math.nan
        overshoot = float((np.max(np.abs(omegas)) - final_ref) / final_ref * 100.0)

    tolerance = 0.02 * max(final_ref, 1.0)
    outside = np.where(abs_error > tolerance)[0]
    settling_time = float(times[outside[-1]]) if outside.size else 0.0

    return RolloutMetrics(
        iae=iae,
        ise=ise,
        steady_state_error=steady_state_error,
        control_energy=control_energy,
        rise_time=rise_time,
        settling_time=settling_time,
        overshoot_pct=overshoot,
        saturation_fraction=saturation_fraction,
    )


def tune_pid_on_nominal() -> tuple[float, float, float]:
    """Small nominal-only grid search; the selected gains are then frozen."""
    kp_values = [0.07, 0.16, 0.30]
    ki_values = [0.4, 1.6, 3.6]
    kd_values = [0.0, 0.0008]
    best_score = float("inf")
    best = (0.10, 1.6, 0.0004)
    tune_scenarios = ["step_nominal"]

    for kp in kp_values:
        for ki in ki_values:
            for kd in kd_values:
                controller = PIDController(kp, ki, kd)
                score = 0.0
                for scenario_name in tune_scenarios:
                    row = rollout_controller(controller, "nominal", scenario_name, seed=123)
                    score += row["iae"] + 0.002 * row["control_energy"] + 0.2 * row["steady_state_error"]
                if score < best_score:
                    best_score = score
                    best = (kp, ki, kd)
    return best


def build_classical_controllers() -> list[Callable[[], BaseController]]:
    params = MotorParams()
    kp, ki, kd = tune_pid_on_nominal()
    return [
        lambda: PIDController(kp, ki, kd),
        lambda: LQRController(params),
        lambda: LQIController(params),
        lambda: MPCLinearClippedController(params),
    ]


def maybe_load_td3(model_path: Path | None) -> Callable[[], BaseController] | None:
    if model_path is None or not model_path.exists():
        return None
    from stable_baselines3 import TD3

    model = TD3.load(str(model_path))
    return lambda: SB3Controller("TD3_loaded", model)


def maybe_load_sac(model_path: Path | None) -> Callable[[], BaseController] | None:
    if model_path is None or not model_path.exists():
        return None
    from stable_baselines3 import SAC

    model = SAC.load(str(model_path))
    return lambda: SB3Controller("SAC_loaded", model)


def maybe_load_ddpg(model_path: Path | None) -> Callable[[], BaseController] | None:
    if model_path is None or not model_path.exists():
        return None
    from stable_baselines3 import DDPG

    model = DDPG.load(str(model_path))
    return lambda: SB3Controller("DDPG_loaded", model)


def train_optional_rl_controller(
    algo_name: str,
    timesteps: int,
    seed: int,
    output_dir: Path,
) -> Callable[[], BaseController]:
    from stable_baselines3 import DDPG, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv

    class ProgressCallback(BaseCallback):
        def __init__(self, interval: int = 5_000):
            super().__init__()
            self.interval = interval
            self.next_report = interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_report:
                print(
                    f"[train:{algo_name}] {self.num_timesteps}/{timesteps} steps",
                    flush=True,
                )
                self.next_report += self.interval
            return True

    def make_env():
        return Monitor(
            DCMotorSpeedEnv(
                condition=get_condition_library()["nominal"],
                scenario_name="step_nominal",
                training_mode=True,
            )
        )

    env = DummyVecEnv([make_env])
    algo_cls = {"DDPG": DDPG, "SAC": SAC}[algo_name]
    kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "seed": seed,
        "verbose": 0,
        "learning_rate": 3e-4,
        "buffer_size": 200_000,
        "batch_size": 256,
        "gamma": 0.995,
        "tau": 0.005,
        "learning_starts": 1_000,
        "train_freq": (10, "step"),
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [64, 64]},
    }
    if algo_name == "DDPG":
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(1, dtype=np.float32),
            sigma=0.15 * np.ones(1, dtype=np.float32),
        )

    model = algo_cls(**kwargs)
    model.learn(total_timesteps=timesteps, progress_bar=False, callback=ProgressCallback())
    model_path = output_dir / f"{algo_name.lower()}_nominal_{timesteps}_steps.zip"
    model.save(str(model_path))
    env.close()
    return lambda: SB3Controller(algo_name, model)


def evaluate_all_controllers(
    controller_factories: list[Callable[[], BaseController]],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    condition_names = list(get_condition_library().keys())
    total = len(controller_factories) * len(condition_names) * len(SCENARIO_NAMES)
    completed = 0
    for factory in controller_factories:
        controller_name = factory().name
        print(f"[evaluate] {controller_name}", flush=True)
        for condition_name in condition_names:
            for scenario_name in SCENARIO_NAMES:
                controller = factory()
                row = rollout_controller(controller, condition_name, scenario_name, seed=seed)
                rows.append(row)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"[progress] {completed}/{total} rollouts", flush=True)
    return pd.DataFrame(rows)


def summarize_results(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby("controller")
        .agg(
            mean_iae=("iae", "mean"),
            std_iae=("iae", "std"),
            mean_ise=("ise", "mean"),
            std_ise=("ise", "std"),
            mean_control_energy=("control_energy", "mean"),
            std_control_energy=("control_energy", "std"),
            mean_steady_state_error=("steady_state_error", "mean"),
            mean_rise_time=("rise_time", "mean"),
            mean_settling_time=("settling_time", "mean"),
            mean_overshoot_pct=("overshoot_pct", "mean"),
            mean_saturation_fraction=("saturation_fraction", "mean"),
        )
        .reset_index()
        .sort_values("mean_iae")
    )

    winner_rows: list[dict] = []
    for (condition, scenario), group in metrics.groupby(["condition", "scenario"]):
        winner_rows.append(
            {
                "condition": condition,
                "scenario": scenario,
                "best_iae_controller": group.loc[group["iae"].idxmin(), "controller"],
                "best_iae": float(group["iae"].min()),
                "best_energy_controller": group.loc[group["control_energy"].idxmin(), "controller"],
                "best_control_energy": float(group["control_energy"].min()),
                "best_sse_controller": group.loc[group["steady_state_error"].idxmin(), "controller"],
                "best_steady_state_error": float(group["steady_state_error"].min()),
            }
        )
    winners = pd.DataFrame(winner_rows).sort_values(["condition", "scenario"])
    return summary, winners


def make_plots(metrics: pd.DataFrame, summary: pd.DataFrame, winners: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics.copy()
    metrics["case"] = metrics["condition"] + " | " + metrics["scenario"]
    pivot = metrics.pivot_table(index="case", columns="controller", values="iae", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index))

    fig_h = max(7.0, 0.36 * len(pivot.index))
    fig_w = max(8.0, 1.45 * len(pivot.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Controller Comparison Heatmap: lower IAE is better")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("IAE")
    fig.tight_layout()
    fig.savefig(output_dir / "controller_comparison_heatmap.png")
    plt.close(fig)

    ordered = summary.sort_values("mean_iae")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=170)
    ax.bar(ordered["controller"], ordered["mean_iae"], yerr=ordered["std_iae"].fillna(0.0), capsize=4)
    ax.set_ylabel("Mean IAE across all cases")
    ax.set_title("Mean IAE by Controller")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_iae_by_controller.png")
    plt.close(fig)

    ordered = summary.sort_values("mean_control_energy")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=170)
    ax.bar(
        ordered["controller"],
        ordered["mean_control_energy"],
        yerr=ordered["std_control_energy"].fillna(0.0),
        capsize=4,
        color="#33658A",
    )
    ax.set_ylabel("Mean control energy")
    ax.set_title("Mean Control Energy by Controller")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_control_energy_by_controller.png")
    plt.close(fig)

    table_cols = ["condition", "scenario", "best_iae_controller", "best_energy_controller", "best_sse_controller"]
    table_df = winners[table_cols].copy()
    fig_h = max(5.5, 0.35 * len(table_df))
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=170)
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)
    ax.set_title("Winning Controller by Test Case", pad=14)
    fig.tight_layout()
    fig.savefig(output_dir / "controller_winner_table.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_output = SCRIPT_DIR / "outputs" / "experiment2_more_controllers"
    default_td3 = (
        SCRIPT_DIR
        / "outputs"
        / "experiment4_condition_aware_10seeds_100k"
        / "seed_00"
        / "td3_model.zip"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--td3-model", type=Path, default=default_td3)
    parser.add_argument("--sac-model", type=Path, default=None)
    parser.add_argument("--ddpg-model", type=Path, default=None)
    parser.add_argument("--no-td3", action="store_true", help="Do not include an existing TD3 model.")
    parser.add_argument("--include-ddpg", action="store_true", help="Train and evaluate DDPG.")
    parser.add_argument("--include-sac", action="store_true", help="Train and evaluate SAC.")
    parser.add_argument("--rl-timesteps", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] output_dir={args.output_dir}", flush=True)
    factories = build_classical_controllers()
    pid = factories[0]()
    if isinstance(pid, PIDController):
        print(f"[setup] tuned PID gains: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}", flush=True)

    if not args.no_td3:
        td3_factory = maybe_load_td3(args.td3_model)
        if td3_factory is not None:
            print(f"[setup] loaded TD3 model: {args.td3_model}", flush=True)
            factories.append(td3_factory)
        else:
            print(f"[setup] TD3 model not found, skipping: {args.td3_model}", flush=True)

    sac_factory = maybe_load_sac(args.sac_model)
    if sac_factory is not None:
        print(f"[setup] loaded SAC model: {args.sac_model}", flush=True)
        factories.append(sac_factory)
    elif args.sac_model is not None:
        print(f"[setup] SAC model not found, skipping: {args.sac_model}", flush=True)

    ddpg_factory = maybe_load_ddpg(args.ddpg_model)
    if ddpg_factory is not None:
        print(f"[setup] loaded DDPG model: {args.ddpg_model}", flush=True)
        factories.append(ddpg_factory)
    elif args.ddpg_model is not None:
        print(f"[setup] DDPG model not found, skipping: {args.ddpg_model}", flush=True)

    if args.include_ddpg:
        print(f"[train] DDPG for {args.rl_timesteps} steps", flush=True)
        factories.append(train_optional_rl_controller("DDPG", args.rl_timesteps, args.seed, args.output_dir))

    if args.include_sac:
        print(f"[train] SAC for {args.rl_timesteps} steps", flush=True)
        factories.append(train_optional_rl_controller("SAC", args.rl_timesteps, args.seed, args.output_dir))

    metrics = evaluate_all_controllers(factories, seed=args.seed)
    summary, winners = summarize_results(metrics)

    metrics_path = args.output_dir / "controller_metrics_long.csv"
    summary_path = args.output_dir / "controller_summary.csv"
    winners_path = args.output_dir / "controller_winner_table.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    winners.to_csv(winners_path, index=False)
    make_plots(metrics, summary, winners, args.output_dir)

    print("[done] wrote:", flush=True)
    print(f"  {metrics_path}", flush=True)
    print(f"  {summary_path}", flush=True)
    print(f"  {winners_path}", flush=True)
    print("[summary]", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
