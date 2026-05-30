from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class MotorParams:
    la: float = 0.58e-3
    ra: float = 2.59
    j: float = 5.69e-4
    bm: float = 1e-6
    kt: float = 28.6e-3
    ke: float = 28.6e-3
    v_max: float = 24.0
    i_max: float = 8.0
    omega_max: float = 150.0


@dataclass(frozen=True)
class PlantCondition:
    name: str
    ra_factor: float = 1.0
    j_factor: float = 1.0
    friction_coulomb: float = 0.0
    saturation_alpha: float = 0.0


def get_condition_library() -> dict[str, PlantCondition]:
    return {
        "nominal": PlantCondition(name="nominal"),
        "ra_plus_50": PlantCondition(name="ra_plus_50", ra_factor=1.5),
        "j_plus_50": PlantCondition(name="j_plus_50", j_factor=1.5),
        "friction": PlantCondition(name="friction", friction_coulomb=0.005),
        "saturation": PlantCondition(name="saturation", saturation_alpha=0.15),
        "combined_stress": PlantCondition(
            name="combined_stress",
            ra_factor=1.3,
            friction_coulomb=0.005,
            saturation_alpha=0.15,
        ),
    }


SCENARIO_NAMES = (
    "step_nominal",
    "step_load_disturbance",
    "ramp",
    "sine",
)

CONDITION_NAMES = (
    "nominal",
    "ra_plus_50",
    "j_plus_50",
    "friction",
    "saturation",
    "combined_stress",
)


class DCMotorSpeedEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        condition: PlantCondition | None = None,
        scenario_name: str | None = None,
        training_mode: bool = True,
        training_condition_mode: str = "nominal",
        training_scenario_mode: str = "uniform",
        reward_profile: str = "balanced",
        sim_dt: float = 1e-4,
        control_dt: float = 1e-3,
        episode_duration: float = 0.5,
    ) -> None:
        super().__init__()
        self.params = MotorParams()
        self.condition = condition or get_condition_library()["nominal"]
        self.training_mode = training_mode
        self.training_condition_mode = training_condition_mode
        self.training_scenario_mode = training_scenario_mode
        if reward_profile not in {"balanced", "tracking_first", "competitive"}:
            raise ValueError(f"Unknown reward_profile: {reward_profile}")
        self.reward_profile = reward_profile
        self.sim_dt = sim_dt
        self.control_dt = control_dt
        self.episode_duration = episode_duration
        self.substeps = int(round(control_dt / sim_dt))
        self.max_steps = int(round(episode_duration / control_dt))
        self.scenario_name = scenario_name or "step_nominal"

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.t = 0.0
        self.state = np.zeros(2, dtype=np.float64)
        self.prev_error = 0.0
        self.prev_reference = 0.0
        self.integral_error = 0.0
        self.prev_voltage = 0.0
        self.load_torque = 0.0
        self.last_error_derivative = 0.0
        self.last_reference_derivative = 0.0
        self.high_voltage_dwell = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "condition" in options:
            self.condition = options["condition"]
        elif self.training_mode and self.training_condition_mode != "nominal":
            self.condition = self._sample_training_condition()

        if options and "scenario_name" in options:
            self.scenario_name = options["scenario_name"]
        elif self.training_mode:
            self.scenario_name = self._sample_training_scenario()

        self.step_count = 0
        self.t = 0.0
        self.state = np.zeros(2, dtype=np.float64)
        reference = self._reference(0.0)
        self.prev_reference = reference
        self.prev_error = reference - self.state[1]
        self.integral_error = 0.0
        self.prev_voltage = 0.0
        self.load_torque = self._load_torque(0.0)
        self.last_error_derivative = 0.0
        self.last_reference_derivative = 0.0
        self.high_voltage_dwell = 0.0
        obs = self._build_obs()
        info = {"scenario_name": self.scenario_name, "condition": self.condition.name}
        return obs, info

    def step(self, action: np.ndarray):
        voltage = float(np.clip(action[0], -1.0, 1.0) * self.params.v_max)

        for _ in range(self.substeps):
            self.load_torque = self._load_torque(self.t)
            self.state = self._rk4_step(self.state, voltage, self.load_torque, self.sim_dt)
            self.t += self.sim_dt

        self.step_count += 1
        reference = self._reference(self.t)
        error = reference - self.state[1]
        self.integral_error += error * self.control_dt
        self.last_error_derivative = (error - self.prev_error) / self.control_dt
        self.last_reference_derivative = (reference - self.prev_reference) / self.control_dt
        current = self.state[0]
        d_voltage = (voltage - self.prev_voltage) / self.control_dt
        truncated = self.step_count >= self.max_steps
        sat_level = self._saturation_level(voltage)
        self.high_voltage_dwell = 0.995 * self.high_voltage_dwell + sat_level

        reward = self._reward(
            error=error,
            current=current,
            voltage=voltage,
            d_voltage=d_voltage,
            sat_level=sat_level,
            is_final=truncated,
        )
        self.prev_error = error
        self.prev_reference = reference
        self.prev_voltage = voltage
        obs = self._build_obs()

        terminated = False
        info = {
            "reference": reference,
            "speed": self.state[1],
            "current": current,
            "voltage": voltage,
            "load_torque": self.load_torque,
            "scenario_name": self.scenario_name,
            "condition": self.condition.name,
        }
        return obs, reward, terminated, truncated, info

    def _sample_training_condition(self) -> PlantCondition:
        conditions = get_condition_library()
        if self.training_condition_mode == "robust":
            weights = np.array([0.18, 0.08, 0.08, 0.08, 0.28, 0.30], dtype=np.float64)
        elif self.training_condition_mode == "hard":
            weights = np.array([0.05, 0.05, 0.05, 0.05, 0.40, 0.40], dtype=np.float64)
        else:
            return conditions["nominal"]
        weights = weights / weights.sum()
        condition_name = self.rng.choice(CONDITION_NAMES, p=weights).item()
        return conditions[condition_name]

    def _sample_training_scenario(self) -> str:
        if self.training_scenario_mode == "hard_tracking":
            weights = np.array([0.32, 0.30, 0.10, 0.28], dtype=np.float64)
            weights = weights / weights.sum()
            return self.rng.choice(SCENARIO_NAMES, p=weights).item()
        return self.rng.choice(SCENARIO_NAMES).item()

    def _reference(self, t: float) -> float:
        if self.scenario_name == "step_nominal":
            return 100.0
        if self.scenario_name == "step_load_disturbance":
            return 100.0
        if self.scenario_name == "ramp":
            return min(200.0 * t, 120.0)
        if self.scenario_name == "sine":
            return 100.0 + 20.0 * np.sin(2.0 * np.pi * 2.0 * t)
        raise ValueError(f"Unknown scenario: {self.scenario_name}")

    def _load_torque(self, t: float) -> float:
        if self.scenario_name == "step_load_disturbance" and t >= 0.2:
            return 0.02
        return 0.0

    def _motor_dynamics(self, state: np.ndarray, voltage: float, load_torque: float) -> np.ndarray:
        ia, omega = state
        ra_eff = self.params.ra * self.condition.ra_factor
        j_eff = self.params.j * self.condition.j_factor
        kt_eff = self.params.kt / (1.0 + self.condition.saturation_alpha * abs(ia))
        tau_f = 0.0
        if omega > 0.0:
            tau_f = self.condition.friction_coulomb
        elif omega < 0.0:
            tau_f = -self.condition.friction_coulomb

        dia = -(ra_eff / self.params.la) * ia - (self.params.ke / self.params.la) * omega
        dia += voltage / self.params.la
        domega = (kt_eff / j_eff) * ia - (self.params.bm / j_eff) * omega
        domega -= load_torque / j_eff
        domega -= tau_f / j_eff
        return np.array([dia, domega], dtype=np.float64)

    def _rk4_step(
        self, state: np.ndarray, voltage: float, load_torque: float, dt: float
    ) -> np.ndarray:
        f = self._motor_dynamics
        k1 = f(state, voltage, load_torque)
        k2 = f(state + 0.5 * dt * k1, voltage, load_torque)
        k3 = f(state + 0.5 * dt * k2, voltage, load_torque)
        k4 = f(state + dt * k3, voltage, load_torque)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[0] = float(np.clip(next_state[0], -self.params.i_max, self.params.i_max))
        next_state[1] = float(np.clip(next_state[1], -self.params.omega_max, self.params.omega_max))
        return next_state

    def _saturation_level(self, voltage: float) -> float:
        return float(np.clip((abs(voltage) / self.params.v_max - 0.85) / 0.15, 0.0, 1.0))

    def _build_obs(self) -> np.ndarray:
        reference = self._reference(self.t)
        error = reference - self.state[1]
        time_fraction = self.step_count / max(1, self.max_steps)
        voltage_sat_level = self._saturation_level(self.prev_voltage)

        obs = np.array(
            [
                self.state[0] / self.params.i_max,
                self.state[1] / self.params.omega_max,
                reference / self.params.omega_max,
                error / self.params.omega_max,
                np.tanh(self.integral_error / 60.0),
                np.clip(self.last_error_derivative / 1000.0, -1.0, 1.0),
                self.load_torque / 0.02,
                np.clip(self.last_reference_derivative / 250.0, -1.0, 1.0),
                time_fraction,
                self.prev_voltage / self.params.v_max,
                voltage_sat_level,
                np.tanh(self.high_voltage_dwell / 40.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _reward(
        self,
        error: float,
        current: float,
        voltage: float,
        d_voltage: float,
        sat_level: float,
        is_final: bool,
    ) -> float:
        abs_error = abs(error)
        e_norm = error / self.params.omega_max
        i_norm = current / self.params.i_max
        u_norm = voltage / self.params.v_max
        du_norm = d_voltage / (2.0 * self.params.v_max / self.control_dt)
        ie_norm = np.tanh(self.integral_error / 60.0)
        is_saturation = self.condition.name == "saturation"
        is_combined = self.condition.name == "combined_stress"
        hard_condition = is_saturation or is_combined
        effective_error = min(abs_error / 60.0, 1.0)
        improving_error = error * self.last_error_derivative < 0.0
        ineffective_high_voltage = sat_level * effective_error
        if improving_error:
            ineffective_high_voltage *= 0.35
        late_episode = max(0.0, (self.step_count / max(1, self.max_steps) - 0.70) / 0.30)

        if is_combined:
            u2_weight, u1_weight = 0.025, 0.010
            sat_weight, ineffective_u_weight, dwell_weight = 0.045, 0.10, 0.006
            terminal_error_weight = 28.0
            late_error_weight = 1.15
        elif is_saturation:
            u2_weight, u1_weight = 0.035, 0.015
            sat_weight, ineffective_u_weight, dwell_weight = 0.060, 0.16, 0.008
            terminal_error_weight = 23.0
            late_error_weight = 0.95
        else:
            u2_weight, u1_weight = 0.060, 0.025
            sat_weight, ineffective_u_weight, dwell_weight = 0.100, 0.28, 0.015
            terminal_error_weight = 12.0
            late_error_weight = 0.55

        tracking_scale = 1.0
        error_scale = 1.0
        late_scale = 1.0
        low_error_gate = float(np.clip((18.0 - abs_error) / 18.0, 0.0, 1.0))
        if self.reward_profile == "tracking_first":
            u2_weight *= 0.25
            u1_weight *= 0.25
            sat_weight *= 0.35
            ineffective_u_weight *= 0.25
            dwell_weight *= 0.40
            terminal_error_weight *= 1.40
            tracking_scale = 1.15
            error_scale = 1.35
            late_scale = 1.35
        elif self.reward_profile == "competitive":
            energy_gate = 0.20 + 0.80 * low_error_gate
            saturation_gate = 0.25 + 0.75 * low_error_gate
            u2_weight *= energy_gate
            u1_weight *= energy_gate
            sat_weight *= saturation_gate
            ineffective_u_weight *= 0.35
            dwell_weight *= saturation_gate
            terminal_error_weight *= 1.75
            tracking_scale = 1.18
            error_scale = 1.22
            late_scale = 1.70

        tracking_bonus = 1.10 * np.exp(-((abs_error / 8.0) ** 2))
        broad_bonus = 0.35 * np.exp(-((abs_error / 35.0) ** 2))
        reward = tracking_scale * (tracking_bonus + broad_bonus)
        reward -= error_scale * 1.80 * abs(e_norm)
        reward -= error_scale * 0.90 * (e_norm ** 2)
        reward -= late_episode * late_scale * late_error_weight * abs(e_norm)
        reward -= 0.18 * abs(ie_norm)
        reward -= u2_weight * (u_norm ** 2)
        reward -= u1_weight * abs(u_norm)
        reward -= 0.04 * (i_norm ** 2)
        reward -= 0.01 * abs(du_norm)
        reward -= sat_weight * (sat_level ** 2)
        reward -= ineffective_u_weight * ineffective_high_voltage
        reward -= dwell_weight * min(self.high_voltage_dwell / 25.0, 1.0)
        if self.reward_profile == "competitive" and abs_error > 12.0:
            underdrive = max(0.0, 0.45 - abs(u_norm)) / 0.45
            reward -= 0.35 * underdrive * min(abs_error / 55.0, 1.0)
        if abs_error <= 2.0:
            reward += 0.25
        elif abs_error <= 5.0:
            reward += 0.10
        if hard_condition and improving_error and abs_error > 10.0:
            reward += 0.08 * sat_level * effective_error
        if is_final:
            reward += 55.0 * np.exp(-((abs_error / 3.5) ** 2))
            reward -= terminal_error_weight * abs(e_norm)
            reward -= 0.02 * min(self.high_voltage_dwell / 25.0, 1.0)
        return float(reward)


def _nominal_motor_state_space(params: MotorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.array(
        [
            [-params.ra / params.la, -params.ke / params.la],
            [params.kt / params.j, -params.bm / params.j],
        ],
        dtype=np.float64,
    )
    b = np.array([[1.0 / params.la], [0.0]], dtype=np.float64)
    c = np.array([[0.0, 1.0]], dtype=np.float64)
    return a, b, c


def _care_gain(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    r_inv = np.linalg.inv(r)
    hamiltonian = np.block([[a, -b @ r_inv @ b.T], [-q, -a.T]])
    eigvals, eigvecs = np.linalg.eig(hamiltonian)
    stable = np.where(np.real(eigvals) < 0.0)[0]
    if stable.size != a.shape[0]:
        stable = np.argsort(np.real(eigvals))[: a.shape[0]]

    v = eigvecs[:, stable]
    v1 = v[: a.shape[0], :]
    v2 = v[a.shape[0] :, :]
    p = np.real(v2 @ np.linalg.inv(v1))
    p = 0.5 * (p + p.T)
    return np.real(r_inv @ b.T @ p)


class ResidualLQRDCMotorEnv(DCMotorSpeedEnv):
    """Environment where the agent learns a residual voltage on top of LQR."""

    def __init__(
        self,
        *args: Any,
        residual_voltage_limit: float = 8.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.residual_voltage_limit = float(residual_voltage_limit)
        a, b, c = _nominal_motor_state_space(self.params)
        q = np.diag([0.003, 2.8])
        r = np.array([[0.045]], dtype=np.float64)
        self._lqr_a = a
        self._lqr_b = b
        self._lqr_c = c
        self._lqr_k = _care_gain(a, b, q, r)
        denom = c @ np.linalg.inv(a - b @ self._lqr_k) @ b
        self._lqr_nbar = float(-1.0 / denom.item())

    def _lqr_voltage(self) -> float:
        x = np.asarray(self.state, dtype=np.float64)
        reference = self._reference(self.t)
        feedback = float((-self._lqr_k @ x.reshape(-1, 1)).item())
        return feedback + self._lqr_nbar * reference

    def step(self, action: np.ndarray):
        residual_normalized = float(np.clip(action[0], -1.0, 1.0))
        baseline_voltage = self._lqr_voltage()
        residual_voltage = residual_normalized * self.residual_voltage_limit
        combined_voltage = float(
            np.clip(
                baseline_voltage + residual_voltage,
                -self.params.v_max,
                self.params.v_max,
            )
        )
        combined_action = np.array([combined_voltage / self.params.v_max], dtype=np.float32)
        obs, reward, terminated, truncated, info = super().step(combined_action)
        info["baseline_voltage"] = baseline_voltage
        info["residual_voltage"] = residual_voltage
        info["combined_voltage"] = combined_voltage
        return obs, reward, terminated, truncated, info
