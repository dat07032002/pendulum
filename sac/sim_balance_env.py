"""
Balance-only Furuta sim, matched to the Nidec hardware for sim-to-real transfer.

This trains a *balance* policy in MuJoCo that can be deployed on the real robot
via run_policy.py --lift-to-catch (you lift the pendulum upright, the policy
catches it). Swing-up is NOT learned here.

Why this transfers (the fidelity that matters, from live-hardware experience):
  - SPEED-controlled arm: u -> arm angular velocity with deadband compensation,
    exactly like the firmware (speed = MIN + |u|*(MAX-MIN)), NOT a torque motor.
  - CABLE-WRAP SPRING: a restoring torque on the arm toward a (randomized)
    neutral -- the disturbance that dominated the real hardware.
  - SENSOR LATENCY + theta_dot filter lag, and a 100 Hz control rate matching
    the firmware/PC loop.
  - DOMAIN RANDOMIZATION over every uncertain quantity, so the policy is robust
    to the sim-to-real gap rather than tuned to one guessed value.

Observation : [cos(theta), sin(theta), theta_dot, phi, phi_dot]  (theta=0 upright)
Action      : scalar u in [-1, 1] (clamped to action_limit), same as hardware.
Reward      : identical to furuta_hw_env (sharp upright + hold bonus + windowed
              theta_dot penalty + arm penalty), so a sim policy optimizes the
              same objective the hardware scores.
Reset       : pendulum starts NEAR upright with some velocity (the catch), arm
              at a random angle. Episode ends when it falls past fall_threshold.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML_PATH = Path(__file__).resolve().parent.parent / "pendulum" / "furuta_pendulum.xml"


class FurutaBalanceSimEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        domain_rand: bool = True,
        episode_seconds: float = 8.0,
        fall_threshold_deg: float = 30.0,
        start_angle_deg: float = 10.0,   # |theta| spread at start (matches 10deg lift-to-catch)
        start_vel: float = 3.0,          # |theta_dot| spread at start (matches handoff |thd|<3)
        action_limit: float = 0.4,
        velocity_control: bool = True,   # True = Nidec speed control; False = torque control
    ):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.model.opt.timestep = 0.002
        self._velocity_control = bool(velocity_control)
        # Implicit integrator: stable for the stiff velocity servo AND the
        # voltage model's back-EMF damping on the arm's tiny inertia.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        if self._velocity_control:
            # Speed-controlled Nidec: reconfigure the shoulder actuator as a
            # torque-limited VELOCITY servo (ctrl = target arm velocity).
            self.model.actuator_gaintype[0] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gear[0, 0] = 1.0
            self.model.actuator_ctrllimited[0] = 1
            self.model.actuator_ctrlrange[0] = [-20.0, 20.0]
            self.model.actuator_forcelimited[0] = 1
        # else: keep the XML torque motor (gear 0.132, ctrl in [-1,1]).
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip   # 0.01 s -> 100 Hz
        self.episode_seconds = float(episode_seconds)
        self.max_steps = round(self.episode_seconds / self.dt)
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        self._domain_rand = bool(domain_rand)
        self._action_limit = float(np.clip(action_limit, 0.0, 1.0))
        self._fall_threshold_rad = np.deg2rad(float(fall_threshold_deg))
        self._start_angle = np.deg2rad(float(start_angle_deg))
        self._start_vel = float(start_vel)

        obs_high = np.array([1.0, 1.0, 70.0, 2.4, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # --- Reward (identical to furuta_hw_env) ---
        self._phi_soft_limit = np.deg2rad(90.0)
        self._phi_boundary_limit = np.deg2rad(100.0)
        self._phi_soft_penalty_gain = 2.5
        self._phi_boundary_penalty_gain = 10.0
        self._upright_reward_width = np.deg2rad(8.0)
        self._theta_dot_penalty_width = np.deg2rad(30.0)
        self._theta_dot_penalty_gain = 0.02
        self._upright_hold_angle = np.deg2rad(10.0)
        self._upright_hold_vel = 3.0
        self._tight_hold_angle = np.deg2rad(5.0)
        self._tight_hold_vel = 1.5

        # --- Nidec speed-control actuator (nominal; DR perturbs these) ---
        # u=1 -> ~10 rad/s arm (we measured ~580 deg/s); deadband floor MIN/MAX=0.24.
        self._nom_max_arm_speed = 10.0     # rad/s at |u|=1
        self._nom_min_frac = 0.24          # MIN_SPEED/MAX_SPEED firmware ratio (deadband comp)
        self._nom_vel_kv = 3.0             # velocity-servo stiffness (high; torque clamp bounds it)
        self._nom_tau_max = 0.13           # motor torque ceiling [N m] (~the model's gear value)
        self._nom_deadband = 0.005         # |u| below this = hold (firmware ACTION_ZERO_ZONE)
        # Voltage / DC-motor model (real Nidec): torque droops with arm speed.
        self._nom_free_speed_max = 10.0    # arm free-run speed at full drive [rad/s] (~580 deg/s)
        self._nom_tau_stall = 0.13         # stall torque at full drive [N m] (uncertain -> wide DR)

        # --- Cable-wrap spring (the hardware nemesis); nominal small, DR widens ---
        self._nom_spring_k = 0.015         # restoring torque per rad of arm wrap [N m/rad]
        self._nom_spring_c = 0.002         # spring damping [N m s/rad]

        # --- Sensor / latency (theta_dot filter + action delay) ---
        self._nom_filter_alpha = 0.5       # matches firmware THETA_VEL_ALPHA
        self._nom_shoulder_damp = float(self.model.dof_damping[0])
        self._nom_arm_mass = float(self.model.body_mass[2])
        self._nom_rod_mass = float(self.model.body_mass[3])
        self._nom_arm_inertia = self.model.body_inertia[2].copy()
        self._nom_rod_inertia = self.model.body_inertia[3].copy()

        self._step_count = 0
        self._balance_mode = False
        self._reset_episode_params()

    # ------------------------------------------------------------------
    def _reset_episode_params(self) -> None:
        """Sample this episode's physical parameters (domain randomization)."""
        rng = getattr(self, "np_random", None)
        p = 1.0 if self._domain_rand else 0.0

        def jitter(nom, frac):
            if p == 0 or rng is None:
                return nom
            return nom * rng.uniform(1 - frac, 1 + frac)

        self._max_arm_speed = jitter(self._nom_max_arm_speed, 0.25)
        self._min_frac = float(np.clip(jitter(self._nom_min_frac, 0.3), 0.1, 0.4))
        self._vel_kv = jitter(self._nom_vel_kv, 0.4)
        self._tau_max = jitter(self._nom_tau_max, 0.3)
        if self._velocity_control:
            self.model.actuator_gainprm[0, 0] = self._vel_kv
            self.model.actuator_biasprm[0, 2] = -self._vel_kv
            self.model.actuator_forcerange[0] = [-self._tau_max, self._tau_max]
        self._free_speed_max = jitter(self._nom_free_speed_max, 0.35)
        # Wide stall-torque DR incl. the weak end -- real authority is uncertain
        # and the hardware looked under-actuated.
        self._tau_stall = self._nom_tau_stall if p == 0 else float(rng.uniform(0.05, 0.20))
        self._deadband = self._nom_deadband if p == 0 else float(rng.uniform(0.0, 0.03))
        self._filter_alpha = self._nom_filter_alpha if p == 0 else float(rng.uniform(0.35, 0.7))

        # Cable spring: from ~0 up to ~3x nominal, with a randomized neutral.
        self._spring_k = self._nom_spring_k if p == 0 else float(rng.uniform(0.0, 3.0 * self._nom_spring_k))
        self._spring_c = self._nom_spring_c if p == 0 else float(rng.uniform(0.0, 2.0 * self._nom_spring_c))
        self._spring_neutral = 0.0 if p == 0 else float(rng.uniform(-np.deg2rad(40.0), np.deg2rad(40.0)))

        # Action latency (control-step delays) + sensor noise scale.
        delay = 0 if p == 0 else int(rng.integers(0, 3))  # 0..2 steps = 0..20 ms
        self._action_buf = collections.deque([0.0] * (delay + 1), maxlen=delay + 1)
        self._theta_pos_noise = 0.0 if p == 0 else float(rng.uniform(0.0, 0.004))
        self._phi_pos_noise = 0.0 if p == 0 else float(rng.uniform(0.0, 0.02))
        self._phi_vel_noise = 0.0 if p == 0 else float(rng.uniform(0.0, 0.08))

        # Mass / damping randomization.
        self.model.dof_damping[0] = jitter(self._nom_shoulder_damp, 0.3)
        self.model.body_mass[2] = self._nom_arm_mass * jitter(1.0, 0.1)
        self.model.body_mass[3] = self._nom_rod_mass * jitter(1.0, 0.1)
        self.model.body_inertia[2] = self._nom_arm_inertia * (self.model.body_mass[2] / self._nom_arm_mass)
        self.model.body_inertia[3] = self._nom_rod_inertia * (self.model.body_mass[3] / self._nom_rod_mass)

    def _nidec_speed(self, u: float) -> float:
        """Firmware u -> commanded arm velocity (deadband-compensated)."""
        if abs(u) < self._deadband:
            return 0.0
        frac = self._min_frac + abs(u) * (1.0 - self._min_frac)
        return float(np.sign(u) * frac * self._max_arm_speed)

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        theta = float(self.data.qpos[1])
        phi = float(self.data.qpos[0])
        phi_dot = float(self.data.qvel[0])
        if self._theta_pos_noise:
            theta += self.np_random.normal(0, self._theta_pos_noise)
        if self._phi_pos_noise:
            phi += self.np_random.normal(0, self._phi_pos_noise)
        if self._phi_vel_noise:
            phi_dot += self.np_random.normal(0, self._phi_vel_noise)

        # theta_dot from a filtered finite difference (AS5600 differentiation).
        raw_dot = (theta - self._prev_theta) / self.dt
        self._theta_dot_filt = (self._filter_alpha * raw_dot
                                + (1.0 - self._filter_alpha) * self._theta_dot_filt)
        self._prev_theta = theta
        return np.array([np.cos(theta), np.sin(theta), self._theta_dot_filt, phi, phi_dot],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_params()
        mujoco.mj_setConst(self.model, self.data)

        rng = self.np_random
        # Start NEAR upright with some velocity -- the catch scenario.
        self.data.qpos[1] = float(rng.uniform(-self._start_angle, self._start_angle))   # elbow ~ upright
        self.data.qvel[1] = float(rng.uniform(-self._start_vel, self._start_vel))
        self.data.qpos[0] = float(rng.uniform(-np.deg2rad(60.0), np.deg2rad(60.0)))      # random arm
        self.data.qvel[0] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._prev_theta = float(self.data.qpos[1])
        self._theta_dot_filt = float(self.data.qvel[1])
        self._step_count = 0
        self._balance_mode = False
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        u = float(np.clip(action[0], -self._action_limit, self._action_limit))
        self._action_buf.append(u)
        u_delayed = self._action_buf[0]
        if self._velocity_control:
            self.data.ctrl[0] = self._nidec_speed(u_delayed)   # velocity-servo tracks this
            drive = 0.0
        else:
            self.data.ctrl[0] = 0.0   # torque comes from the voltage model below
            if abs(u_delayed) < self._deadband:
                drive = 0.0
            else:
                drive = float(np.sign(u_delayed) * (self._min_frac + abs(u_delayed) * (1.0 - self._min_frac)))

        for _ in range(self.frame_skip):
            phi = float(self.data.qpos[0])
            phi_dot = float(self.data.qvel[0])
            tau_spring = -self._spring_k * (phi - self._spring_neutral) - self._spring_c * phi_dot
            if self._velocity_control:
                self.data.qfrc_applied[0] = tau_spring
            else:
                # DC-motor / voltage model: torque high when slow, dropping to
                # zero at the free-run speed (back-EMF). Matches the grab test
                # and the under-actuation seen on hardware.
                tau_motor = self._tau_stall * (drive - phi_dot / self._free_speed_max)
                tau_motor = float(np.clip(tau_motor, -self._tau_stall, self._tau_stall))
                self.data.qfrc_applied[0] = tau_motor + tau_spring
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        theta = float(self.data.qpos[1])
        theta = (theta + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
        phi = float(self.data.qpos[0])
        theta_dot = float(obs[2])
        angle_error = abs(theta)

        balance = np.cos(theta)
        ctrl_cost = -0.1 * u ** 2
        excess = max(0.0, abs(phi) - self._phi_soft_limit)
        boundary_excess = max(0.0, abs(phi) - self._phi_boundary_limit)
        penalty = (-self._phi_soft_penalty_gain * excess ** 2
                   - self._phi_boundary_penalty_gain * boundary_excess ** 2)
        upright_reward = 2.0 * np.exp(-((angle_error / self._upright_reward_width) ** 2))
        theta_dot_window = np.exp(-((angle_error / self._theta_dot_penalty_width) ** 2))
        theta_dot_cost = -self._theta_dot_penalty_gain * theta_dot ** 2 * theta_dot_window
        upright_hold = angle_error < self._upright_hold_angle and abs(theta_dot) < self._upright_hold_vel
        tight_hold = angle_error < self._tight_hold_angle and abs(theta_dot) < self._tight_hold_vel
        hold_reward = (3.0 if upright_hold else 0.0) + (5.0 if tight_hold else 0.0)

        terminated = False
        if angle_error <= np.deg2rad(10.0):
            self._balance_mode = True
        elif self._balance_mode and angle_error > self._fall_threshold_rad:
            terminated = True

        reward = float(balance + ctrl_cost + penalty + upright_reward + theta_dot_cost + hold_reward)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self._render_human()
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def _render_human(self):
        import mujoco.viewer
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = FurutaBalanceSimEnv(domain_rand=True)
    print("Running gymnasium env checker...")
    check_env(env, warn=True)
    print("check_env passed.\n")
    print(f"control dt : {env.dt*1000:.1f} ms ({1/env.dt:.0f} Hz), max_steps={env.max_steps}")

    # Hold-still baseline: does a near-upright start survive a few steps with u=0?
    obs, _ = env.reset(seed=0)
    print(f"start: theta={np.degrees(np.arctan2(obs[1], obs[0])):+.1f}deg, "
          f"phi={np.degrees(obs[3]):+.1f}deg")
    total, steps = 0.0, 0
    for _ in range(env.max_steps):
        obs, rew, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        total += rew; steps += 1
        if term or trunc:
            break
    print(f"u=0 from near-upright: {steps} steps before fall, reward={total:.1f} "
          f"(expect a quick fall with no control)")
