"""
Gymnasium environment wrapping the real Furuta pendulum over the ESP32 serial bridge.

Protocol (matches hardware_smoke_test.py / hardware_policy_limited.py):
  ESP32 -> PC : lines containing  obs=[cos_theta,sin_theta,theta_dot,phi,phi_dot]
  PC -> ESP32 : "u <float>\n"

step() sends the action, then waits one control period while draining the serial
buffer and keeps the freshest observation. The reward replicates furuta_env.py
so a policy trained here is comparable to the sim runs.

Safety:
  - "u 0" is sent on reset, on every termination, and in close().
  - Episode terminates (with a penalty) if |phi| exceeds the angle limit,
    or if no valid observation arrives within obs_timeout.
  - reset() waits for the pendulum to settle hanging before the next episode.
"""

from __future__ import annotations

import re
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import serial

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def parse_obs(line: str) -> np.ndarray | None:
    match = OBS_RE.search(line)
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) != 5:
        return None
    try:
        return np.array([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None


class FurutaHardwareEnv(gym.Env):
    """Real-hardware Furuta pendulum. One instance owns the serial port."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: str = "COM5",
        baud: int = 115200,
        control_dt: float = 0.02,
        episode_seconds: float = 15.0,
        action_limit: float = 1.0,
        phi_limit_deg: float = 90.0,
        obs_timeout: float = 0.3,
        settle_seconds: float = 1.0,
        settle_max_wait: float = 25.0,
        safety_penalty: float = 50.0,
        fall_threshold_deg: float = 20.0,
        recenter: bool = True,
        recenter_u: float = 0.28,
        recenter_u_far: float = 0.40,
        recenter_tol_deg: float = 8.0,
        recenter_max_wait: float = 30.0,
    ):
        super().__init__()
        obs_high = np.array([1.0, 1.0, 55.0, 2.5, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.control_dt = float(control_dt)
        self.max_steps = round(float(episode_seconds) / self.control_dt)
        self._action_limit = float(np.clip(action_limit, 0.0, 1.0))
        self._phi_limit = np.radians(phi_limit_deg)
        self._obs_timeout = float(obs_timeout)
        self._settle_seconds = float(settle_seconds)
        self._settle_max_wait = float(settle_max_wait)
        self._safety_penalty = float(safety_penalty)
        self._fall_threshold_rad = np.deg2rad(float(fall_threshold_deg))
        self._phi_soft_limit = np.deg2rad(60.0)  # penalty before the hardware stop
        self._recenter = bool(recenter)
        self._recenter_u = abs(float(recenter_u))
        self._recenter_u_far = abs(float(recenter_u_far))
        self._recenter_tol = np.deg2rad(float(recenter_tol_deg))
        self._recenter_max_wait = float(recenter_max_wait)
        self._recenter_ok = np.deg2rad(20.0)  # episode may not start beyond this

        self._step_count = 0
        self._balance_mode = False
        self._last_obs = np.zeros(5, dtype=np.float32)
        # Gravity-based zero trim: a settled hanging pendulum is at exactly
        # +/-pi, so any residual from the boot-time upright calibration is
        # measured during reset() and subtracted from theta here.
        self._theta_trim = 0.0

        self._port = serial.Serial(port, baud, timeout=0.005)
        time.sleep(2.0)  # ESP32 resets on serial open
        self._port.reset_input_buffer()
        self._send_u(0.0)

    # ------------------------------------------------------------------
    def _send_u(self, u: float) -> None:
        self._port.write(f"u {u:.5f}\n".encode("ascii"))
        self._port.flush()

    def _read_latest_obs(self, deadline: float) -> np.ndarray | None:
        """Drain serial until the deadline; return the freshest valid obs (or None)."""
        latest = None
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            raw = self._port.readline()
            if not raw:
                continue
            obs = parse_obs(raw.decode("utf-8", errors="replace").strip())
            if obs is not None:
                latest = obs
        return latest

    def _apply_theta_trim(self, obs: np.ndarray) -> np.ndarray:
        if self._theta_trim == 0.0:
            return obs
        theta = np.arctan2(obs[1], obs[0]) - self._theta_trim
        obs = obs.copy()
        obs[0] = np.cos(theta)
        obs[1] = np.sin(theta)
        return obs

    def _wait_for_obs(self) -> np.ndarray | None:
        """Block until one valid obs arrives or obs_timeout expires."""
        deadline = time.perf_counter() + self._obs_timeout
        while time.perf_counter() < deadline:
            raw = self._port.readline()
            if not raw:
                continue
            obs = parse_obs(raw.decode("utf-8", errors="replace").strip())
            if obs is not None:
                return obs
        return None

    def _brake(self, max_seconds: float = 2.0) -> None:
        """Actively drive against the arm's spin until it is slow.

        Cutting power alone lets the low-friction gearbox coast 100+ degrees
        from speed; opposing the velocity stops it within a few control
        periods. Exits immediately if the arm is already slow.
        """
        deadline = time.perf_counter() + max_seconds
        while time.perf_counter() < deadline:
            obs = self._wait_for_obs()
            if obs is None:
                break
            phi_dot = float(obs[4])
            if abs(phi_dot) < 0.8:
                break
            self._send_u(-np.sign(phi_dot) * self._recenter_u)
            time.sleep(0.02)
        self._send_u(0.0)

    def _recenter_arm(self) -> None:
        """Brake, then drive the arm back near phi=0 with short pulses.

        Pulse-and-coast keeps speeds low despite the motor deadband: a brief
        push above the deadband, then coast while the arm decelerates, repeat.
        Far from center a stronger pulse is used so large displacements
        (e.g. after a coasting overshoot) still get home within the budget.
        The pendulum will swing during this; the settle phase afterwards
        handles that.
        """
        self._brake()
        start = time.perf_counter()
        while time.perf_counter() - start < self._recenter_max_wait:
            obs = self._wait_for_obs()
            if obs is None:
                break
            phi, phi_dot = float(obs[3]), float(obs[4])
            if abs(phi) <= self._recenter_tol and abs(phi_dot) < 0.5:
                break
            # Allowed approach speed shrinks with distance so the arm cannot
            # cross center fast and overshoot past the angle limits.
            speed_cap = min(2.5, 0.5 + 1.5 * abs(phi))
            moving_away = phi * phi_dot > 0
            if moving_away and abs(phi_dot) > 0.5:
                self._send_u(-np.sign(phi_dot) * self._recenter_u)   # brake outward motion
            elif abs(phi_dot) > speed_cap:
                self._send_u(-np.sign(phi_dot) * self._recenter_u)   # brake: approaching too fast
            elif abs(phi) > self._recenter_tol and abs(phi_dot) < 0.5 * speed_cap:
                u = self._recenter_u if abs(phi) < self._recenter_ok else self._recenter_u_far
                self._send_u(-np.sign(phi) * u)                      # push toward center
            else:
                self._send_u(0.0)                                    # comfortable speed: coast
            time.sleep(0.02)
        self._send_u(0.0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._send_u(0.0)

        manual = not self._recenter
        if self._recenter:
            for attempt in range(1, 4):
                self._recenter_arm()
                check = self._wait_for_obs()
                if check is not None and abs(float(check[3])) <= self._recenter_ok:
                    break
                phi_str = "no obs" if check is None else f"phi={np.degrees(float(check[3])):+.1f}deg"
                print(f"Recenter attempt {attempt} incomplete ({phi_str}); retrying.")
            else:
                self._send_u(0.0)
                raise RuntimeError(
                    "Arm failed to recenter after 3 attempts; motor stopped. "
                    "Check the arm, cable, and motor before resuming."
                )
        else:
            print("\a>>> Recenter the arm by hand, then let the pendulum hang still...", flush=True)

        # Wait for a valid start pose: pendulum hanging still (cos(theta) ~ -1,
        # low velocities) and, in manual mode, the arm hand-centered. Manual
        # mode waits for the user indefinitely with periodic reminders.
        start = time.perf_counter()
        settled_since = None
        last_reminder = start
        obs = None
        while True:
            now = time.perf_counter()
            if not manual and now - start >= self._settle_max_wait:
                break
            self._send_u(0.0)  # keep refreshing the stop command
            candidate = self._wait_for_obs()
            if candidate is None:
                continue
            obs = candidate
            hanging = candidate[0] < -0.95
            quiet = abs(candidate[2]) < 0.3 and abs(candidate[4]) < 0.3
            centered = abs(float(candidate[3])) <= self._recenter_ok
            ready = hanging and quiet and (centered or not manual)
            now = time.perf_counter()
            if ready:
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= self._settle_seconds:
                    break
            else:
                settled_since = None
                if manual and now - last_reminder >= 5.0:
                    print(f">>> still waiting (phi={np.degrees(float(candidate[3])):+.1f}deg, "
                          f"cos_theta={candidate[0]:+.2f})...", flush=True)
                    last_reminder = now

        if obs is None:
            raise RuntimeError("No observations from ESP32 during reset; check the serial link.")

        # Settled hanging pose is exactly +/-pi: measure the residual zero error.
        if settled_since is not None:
            trim_samples = []
            deadline = time.perf_counter() + 0.3
            while time.perf_counter() < deadline:
                sample = self._wait_for_obs()
                if sample is not None:
                    trim_samples.append(sample)
            if trim_samples:
                arr = np.array(trim_samples)
                raw_theta = float(np.arctan2(arr[:, 1].mean(), arr[:, 0].mean()))
                self._theta_trim = (raw_theta - np.pi + np.pi) % (2 * np.pi) - np.pi
                obs = trim_samples[-1]

        obs = self._apply_theta_trim(obs)
        if manual:
            print(f"\aEpisode starting: phi={np.degrees(float(obs[3])):+.1f}deg", flush=True)
        self._step_count = 0
        self._balance_mode = False
        self._last_obs = obs
        self._next_tick = time.perf_counter() + self.control_dt
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        u = float(np.clip(action[0], -self._action_limit, self._action_limit))
        self._send_u(u)

        obs = self._read_latest_obs(self._next_tick)
        self._next_tick += self.control_dt
        if obs is None:
            obs = self._wait_for_obs()
            self._next_tick = time.perf_counter() + self.control_dt
        if obs is None:
            self._send_u(0.0)
            print("Safety stop: serial observation timeout.")
            return self._last_obs, -self._safety_penalty, True, False, {"safety_stop": "obs_timeout"}
        obs = self._apply_theta_trim(obs)
        self._last_obs = obs

        cos_t, sin_t, theta_dot, phi, phi_dot = (float(v) for v in obs)

        # Hardware safety limit -> brake the arm, stop motor, end episode.
        if abs(phi) > self._phi_limit:
            self._brake()
            print(f"Safety stop: phi_limit (phi={np.degrees(phi):+.1f}deg, phi_dot={phi_dot:+.2f})")
            return obs, -self._safety_penalty, True, False, {"safety_stop": "phi_limit"}

        # Reward identical in shape to furuta_env.py (theta=0 is upright).
        theta = float(np.arctan2(sin_t, cos_t))
        balance = cos_t
        ctrl_cost = -0.1 * u ** 2
        excess = max(0.0, abs(phi) - self._phi_soft_limit)
        penalty = -0.5 * excess ** 2

        angle_error = abs(theta)
        upright_bonus = 0.0
        theta_dot_cost = 0.0
        terminated = False
        if angle_error <= np.deg2rad(10.0):
            self._balance_mode = True
            upright_bonus += 0.2
            theta_dot_cost = -0.005 * theta_dot ** 2
            if angle_error <= np.deg2rad(5.0):
                upright_bonus += 0.2
        elif self._balance_mode and angle_error > self._fall_threshold_rad:
            terminated = True

        reward = float(balance + ctrl_cost + penalty + upright_bonus + theta_dot_cost)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        if terminated or truncated:
            self._send_u(0.0)

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def close(self):
        try:
            if self._port.is_open:
                self._send_u(0.0)
                time.sleep(0.05)
                self._send_u(0.0)
                self._port.close()
        except Exception:
            pass
