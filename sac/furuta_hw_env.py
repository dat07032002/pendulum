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
        phi_limit_deg: float = 120.0,       # episode stop = firmware backstop (intended full range)
        obs_timeout: float = 0.3,
        settle_seconds: float = 1.0,
        settle_max_wait: float = 25.0,
        safety_penalty: float = 50.0,
        fall_threshold_deg: float = 20.0,
        recenter: bool = True,
        recenter_u: float = 0.15,           # servo speed cap (low: motor is fast)
        recenter_kp: float = 0.5,           # servo P gain (u per rad)
        recenter_kd: float = 0.18,          # servo D gain (damps overshoot)
        recenter_tol_deg: float = 10.0,
        recenter_max_wait: float = 18.0,
        swingup_reset: bool = False,        # energy swing-up in reset() -> balance-only RL
        swingup_handoff_deg: float = 30.0,  # hand off to the policy within this of upright
        swingup_gain: float = 0.5,
        swingup_u: float = 0.28,
        swingup_omega2: float = 100.0,
        swingup_timeout: float = 12.0,
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
        self._phi_soft_limit = np.deg2rad(90.0)   # soft penalty starts here
        self._phi_boundary_limit = np.deg2rad(100.0)  # steeper ramp before the hard stop
        self._phi_soft_penalty_gain = 2.5         # 5x stronger (was 0.5)
        self._phi_boundary_penalty_gain = 10.0    # 5x stronger (was 2.0)
        self._upright_reward_width = np.deg2rad(8.0)
        self._theta_dot_penalty_width = np.deg2rad(30.0)
        self._theta_dot_penalty_gain = 0.02
        self._upright_hold_angle = np.deg2rad(10.0)
        self._upright_hold_vel = 3.0
        self._tight_hold_angle = np.deg2rad(5.0)
        self._tight_hold_vel = 1.5
        self._recenter = bool(recenter)
        self._recenter_u = abs(float(recenter_u))      # servo speed cap (U_MAX)
        self._recenter_kp = abs(float(recenter_kp))    # servo P gain
        self._recenter_kd = abs(float(recenter_kd))    # servo D gain (damping)
        self._recenter_tol = np.deg2rad(float(recenter_tol_deg))
        self._recenter_max_wait = float(recenter_max_wait)
        self._recenter_ok = np.deg2rad(20.0)  # episode may not start beyond this

        # Two-controller mode: energy swing-up in reset() -> RL trains balance only.
        self._swingup_reset = bool(swingup_reset)
        self._swingup_handoff = np.deg2rad(float(swingup_handoff_deg))
        self._swingup_gain = float(swingup_gain)
        self._swingup_u = abs(float(swingup_u))
        self._swingup_omega2 = float(swingup_omega2)
        self._swingup_timeout = float(swingup_timeout)

        self._step_count = 0
        self._balance_mode = False
        self._last_obs = np.zeros(5, dtype=np.float32)
        self._episode_upright_steps = 0
        self._episode_upright_streak = 0
        self._episode_best_hold_steps = 0
        self._episode_phi_limit_stops = 0
        # Gravity-based zero trim: a settled hanging pendulum is at exactly
        # +/-pi, so any residual from the boot-time upright calibration is
        # measured during reset() and subtracted from theta here.
        self._theta_trim = 0.0
        # Frozen-theta detector (AS5600 cable disconnect: firmware holds the
        # last value, so cos/sin theta go exactly constant). Only count this
        # while the system is being driven or the arm is moving; a still,
        # quantized pendulum can otherwise look exactly unchanged.
        self._frozen_steps = 0
        self._frozen_limit = 35   # 0.7s; the theta_should_change guard avoids false positives
        self._last_cos = 2.0   # impossible value -> first step never matches
        self._last_sin = 2.0
        # Cable-adaptive recenter: re-zero phi at the arm's natural rest each
        # episode (work WITH the spring, not against it). Track net winding.
        self._cumulative_wind = 0.0
        self._wind_warn_rad = np.deg2rad(70.0)

        self._port = serial.Serial(port, baud, timeout=0.005)
        time.sleep(2.0)  # ESP32 resets on serial open
        self._port.reset_input_buffer()
        self._send_u(0.0)

    # ------------------------------------------------------------------
    def _send_u(self, u: float) -> None:
        self._port.write(f"u {u:.5f}\n".encode("ascii"))
        self._port.flush()

    def _send_zero(self) -> None:
        """Zero the arm encoder (phi=0) at its current position."""
        self._port.write(b"z\n")
        self._port.flush()

    def _swing_up(self) -> np.ndarray:
        """Energy-pump the pendulum to near upright, then hand off to the policy.

        Controller 1 of the two-controller architecture: classical energy
        pumping (no model needed). Returns the raw obs once |theta| reaches the
        hand-off band; on timeout returns the last obs (episode starts wherever
        it is). theta=0 is upright; cos/sin theta are obs[0:2].
        """
        start = time.perf_counter()
        last = self._last_obs
        while time.perf_counter() - start < self._swingup_timeout:
            obs = self._wait_for_obs()
            if obs is None:
                continue
            last = obs
            cos_t, sin_t, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
            theta = float(np.arctan2(sin_t, cos_t))
            if abs(theta) < self._swingup_handoff:
                return obs
            energy = 0.5 * theta_dot ** 2 + self._swingup_omega2 * (cos_t - 1.0)
            direction = 1.0 if (theta_dot * cos_t) > 0 else -1.0
            u = float(np.clip(self._swingup_gain * (-energy) * direction,
                              -self._swingup_u, self._swingup_u))
            self._send_u(u)
            time.sleep(self.control_dt)
        self._send_u(0.0)
        return last

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

    def _episode_info(self, safety_stop: str | None = None) -> dict:
        info = {
            "upright_steps": self._episode_upright_steps,
            "best_hold": self._episode_best_hold_steps * self.control_dt,
            "best_hold_steps": self._episode_best_hold_steps,
            "phi_limit_stops": self._episode_phi_limit_stops,
        }
        if safety_stop:
            info["safety_stop"] = safety_stop
        return info

    def _brake(self, max_seconds: float = 1.5) -> None:
        """Stop the arm and wait until it is slow.

        The Nidec is speed-controlled: u=0 sets a zero-speed setpoint (active
        stop), and the cable-wrap spring + firmware brake decelerate the arm.
        No counter-drive needed (that was for the old coasting gearbox).
        """
        deadline = time.perf_counter() + max_seconds
        self._send_u(0.0)
        while time.perf_counter() < deadline:
            obs = self._wait_for_obs()
            if obs is None:
                break
            if abs(float(obs[4])) < 0.8:
                break
            self._send_u(0.0)
            time.sleep(0.02)
        self._send_u(0.0)

    def _recenter_arm(self) -> None:
        """Velocity-servo the arm to phi=0 (Nidec speed control + cable spring).

        u = clamp(-Kp*phi, -U_MAX, U_MAX) drives toward center; the cable-wrap
        spring assists by pulling the arm toward neutral. U_MAX is kept low
        because the motor is fast (it would otherwise overshoot center).
        Settles when within tolerance and slow for a short hold.
        """
        self._brake()
        start = time.perf_counter()
        settled_since = None
        while time.perf_counter() - start < self._recenter_max_wait:
            obs = self._wait_for_obs()
            if obs is None:
                break
            phi, phi_dot = float(obs[3]), float(obs[4])
            if abs(phi) <= self._recenter_tol and abs(phi_dot) < 0.5:
                if settled_since is None:
                    settled_since = time.perf_counter()
                elif time.perf_counter() - settled_since >= 0.3:
                    break
            else:
                settled_since = None
            u = float(np.clip(-self._recenter_kp * phi - self._recenter_kd * phi_dot,
                              -self._recenter_u, self._recenter_u))
            self._send_u(u)
            time.sleep(self.control_dt)
        self._send_u(0.0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._send_u(0.0)

        manual = not self._recenter
        if manual:
            print("\a>>> Recenter the arm by hand, then let the pendulum hang still...", flush=True)
        # Auto mode adapts to the cable spring: don't fight it to phi=0. Cut the
        # motor and let the spring settle the arm to its natural rest, then
        # re-zero phi there (below). No servo -> nothing to oscillate against.

        start = time.perf_counter()
        settled_since = None
        last_reminder = start
        obs = None
        while True:
            now = time.perf_counter()
            if not manual and now - start >= self._settle_max_wait:
                break
            self._send_u(0.0)  # motor off; spring settles the arm (auto) / user holds (manual)
            candidate = self._wait_for_obs()
            if candidate is None:
                continue
            obs = candidate
            phi = float(candidate[3])
            hanging = candidate[0] < -0.95
            pend_quiet = abs(candidate[2]) < 0.3
            arm_still = abs(candidate[4]) < 0.5
            if manual:
                ready = hanging and pend_quiet and arm_still and abs(phi) <= self._recenter_ok
            else:
                ready = hanging and pend_quiet and arm_still  # accept the natural rest position
            if ready:
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= self._settle_seconds:
                    break
            else:
                settled_since = None
                if manual and now - last_reminder >= 5.0:
                    print(f">>> still waiting (phi={np.degrees(phi):+.1f}deg, "
                          f"cos_theta={candidate[0]:+.2f})...", flush=True)
                    last_reminder = now

        if obs is None:
            raise RuntimeError("No observations from ESP32 during reset; check the serial link.")

        # Cable-adaptive: re-zero phi at the natural rest so each episode starts
        # centered relative to the cable's current neutral. Track net winding.
        if not manual:
            self._cumulative_wind += float(obs[3])
            self._send_zero()
            time.sleep(0.1)
            self._port.reset_input_buffer()
            if abs(self._cumulative_wind) > self._wind_warn_rad:
                print(f"WARNING: cable wound ~{np.degrees(self._cumulative_wind):+.0f}deg net; "
                      f"unwind the arm by hand if drift continues.")
            z = self._wait_for_obs()
            if z is not None:
                obs = z

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

        # Controller 1: swing the pendulum up to near upright, then the episode
        # (controller 2, the RL policy) trains purely on the balance/catch.
        if self._swingup_reset:
            print(">>> swing-up...", flush=True)
            obs = self._swing_up()
            print(f">>> hand-off: theta={np.degrees(float(np.arctan2(obs[1], obs[0]))):+.1f}deg",
                  flush=True)

        obs = self._apply_theta_trim(obs)
        bell = "\a" if manual else ""
        print(f"{bell}Episode starting: phi={np.degrees(float(obs[3])):+.1f}deg", flush=True)
        self._step_count = 0
        self._balance_mode = False
        self._episode_upright_steps = 0
        self._episode_upright_streak = 0
        self._episode_best_hold_steps = 0
        self._episode_phi_limit_stops = 0
        self._frozen_steps = 0
        self._last_cos, self._last_sin = float(obs[0]), float(obs[1])
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
            return self._last_obs, -self._safety_penalty, True, False, self._episode_info("obs_timeout")
        obs = self._apply_theta_trim(obs)
        self._last_obs = obs

        cos_t, sin_t, theta_dot, phi, phi_dot = (float(v) for v in obs)

        # Frozen-theta detector: firmware holds the last good AS5600 value on
        # read failures/spike rejection. Only treat exact repeats as suspicious
        # when there is motor command or arm motion; otherwise a settled
        # pendulum can quantize to the same angle for a while.
        theta_exactly_same = cos_t == self._last_cos and sin_t == self._last_sin
        theta_should_change = abs(u) > 0.02 or abs(phi_dot) > 0.5
        if theta_exactly_same and theta_should_change:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0
        self._last_cos, self._last_sin = cos_t, sin_t
        if self._frozen_steps >= self._frozen_limit:
            self._send_u(0.0)
            print("Safety stop: theta sensor frozen (check AS5600 cable).")
            return obs, -self._safety_penalty, True, False, self._episode_info("sensor_frozen")

        # Hardware safety limit -> brake the arm, stop motor, end episode.
        if abs(phi) > self._phi_limit:
            self._episode_phi_limit_stops += 1
            self._brake()
            print(f"Safety stop: phi_limit (phi={np.degrees(phi):+.1f}deg, phi_dot={phi_dot:+.2f})")
            return obs, -self._safety_penalty, True, False, self._episode_info("phi_limit")

        # Hardware reward: broad swing-up term plus sharper catch terms near
        # upright. The narrow Gaussian fixes the old nearly-flat top reward,
        # while the wider velocity window teaches the pendulum to slow down
        # before it sails through upright.
        theta = float(np.arctan2(sin_t, cos_t))
        balance = cos_t
        ctrl_cost = -0.1 * u ** 2
        excess = max(0.0, abs(phi) - self._phi_soft_limit)
        boundary_excess = max(0.0, abs(phi) - self._phi_boundary_limit)
        penalty = -self._phi_soft_penalty_gain * excess ** 2 - self._phi_boundary_penalty_gain * boundary_excess ** 2

        angle_error = abs(theta)
        upright_reward = 2.0 * np.exp(-((angle_error / self._upright_reward_width) ** 2))
        theta_dot_window = np.exp(-((angle_error / self._theta_dot_penalty_width) ** 2))
        theta_dot_cost = -self._theta_dot_penalty_gain * theta_dot ** 2 * theta_dot_window
        upright_hold = angle_error < self._upright_hold_angle and abs(theta_dot) < self._upright_hold_vel
        tight_hold = angle_error < self._tight_hold_angle and abs(theta_dot) < self._tight_hold_vel
        hold_reward = (3.0 if upright_hold else 0.0) + (5.0 if tight_hold else 0.0)
        if upright_hold:
            self._episode_upright_steps += 1
            self._episode_upright_streak += 1
            self._episode_best_hold_steps = max(self._episode_best_hold_steps, self._episode_upright_streak)
        else:
            self._episode_upright_streak = 0
        terminated = False
        if angle_error <= np.deg2rad(10.0):
            self._balance_mode = True
        elif self._balance_mode and angle_error > self._fall_threshold_rad:
            terminated = True

        reward = float(balance + ctrl_cost + penalty + upright_reward + theta_dot_cost + hold_reward)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        if terminated or truncated:
            self._send_u(0.0)

        info = self._episode_info() if terminated or truncated else {}
        return obs, reward, terminated, truncated, info

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
