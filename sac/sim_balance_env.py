"""
Balance-only Furuta sim, matched to the Nidec hardware for sim-to-real transfer.

This trains a *balance* policy in MuJoCo that can be deployed on the real robot
via run_policy.py --lift-to-catch (you lift the pendulum upright, the policy
catches it). Swing-up is NOT learned here.

Why this transfers (the fidelity that matters, from live-hardware experience):
  - SPEED-controlled arm: u -> arm angular velocity with deadband compensation,
    exactly like the firmware (speed = MIN + |u|*(MAX-MIN)), NOT a torque motor.
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

from energy_swingup import EnergySwingUp

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
        action_limit: float = 1.0,
        velocity_control: bool = True,   # True = Nidec speed control; False = torque control
        curriculum: bool = False,        # ramp difficulty (arrival speed/angle + DR) easy->hard
        corridor_reset: bool = True,     # sample starts from the energy-gated handoff corridor
        switch_dE: float = 0.08,         # energy tolerance (fraction of E_max) for the corridor
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
        # Curriculum: progress 0 (easy) -> 1 (full difficulty). Ramps arrival
        # speed/angle and DR strength; set during training via set_progress().
        self._curriculum = bool(curriculum)
        self._curriculum_progress = 1.0
        self._start_vel_floor = 1.0            # rad/s at progress 0 (near-rest)
        self._start_angle_floor = np.deg2rad(4.0)

        # --- Region-of-attraction reset: sample the energy-gated corridor ---
        # The deployment runner only hands off states with |theta| < switch_in and
        # |E - E_upright| < switch_dE*E_max. Sampling that same corridor (instead of
        # an independent theta x theta_dot rectangle) keeps training/eval starts on
        # physically catchable states rather than uncatchable super-energetic corners.
        self._corridor_reset = bool(corridor_reset)
        self._switch_dE = float(switch_dE)
        self._dE_floor_frac = 0.3              # curriculum floor for the energy tolerance
        self._m_g_lcm = EnergySwingUp.M_ROD * EnergySwingUp.G * EnergySwingUp.L_CM
        self._E_ref = self._m_g_lcm            # upright-rest mechanical energy
        self._E_max = 2.0 * self._m_g_lcm      # hanging->upright energy gap
        self._I_rod = EnergySwingUp.I_ROD

        # obs = [cos(theta), sin(theta), theta_dot, phi, phi_dot, u_prev]
        obs_high = np.array([1.0, 1.0, 70.0, 2.4, 20.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._last_u = 0.0                     # previous commanded action (obs channel)

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
        # Arm-velocity penalty (windowed near upright so the catch can still swing
        # the arm hard) and an action-smoothness penalty (anti-chatter, uses u_prev).
        self._phi_dot_penalty_gain = 0.002
        self._action_smooth_gain = 0.05

        # --- Nidec speed-control actuator (nominal; DR perturbs these) ---
        # sysid Phase 1 (terminal velocity sweep): at u=0.30, phi_dot_ss ~12.8 rad/s.
        # Grounded by sysid at MAX_SPEED=0.40 (re-run after the 0.25->0.40 bump):
        #   slope (free-speed) ~54-60 rad/s; gear=0.035 N·m at I_arm=1e-4, but
        #   I_arm is ~1.6-2e-4 (arm+rotor) so real stall torque ~0.055. Wide DR.
        self._nom_max_arm_speed = 55.0     # rad/s at |u|=1 (sysid slope, backstop-corrected)
        self._nom_min_frac = 0.15          # MIN_SPEED/MAX_SPEED = 0.06/0.40
        self._nom_vel_kv = 3.0             # velocity-servo stiffness (velocity mode only)
        self._nom_tau_max = 0.055          # motor torque ceiling [N m] (velocity mode)
        self._nom_deadband = 0.05          # |u| below this = hold (firmware ACTION_ZERO_ZONE = 5%)
        # Voltage / DC-motor model (real Nidec): torque droops with arm speed.
        self._nom_free_speed_max = 55.0    # arm free-run speed at full drive [rad/s] (sysid @0.40)
        self._nom_tau_stall = 0.055        # stall torque [N m] (gear 0.035 * I_arm correction)

        # --- Sensor / latency (theta_dot filter + action delay) ---
        self._nom_filter_alpha = 0.5       # matches firmware THETA_VEL_ALPHA
        # The voltage model's back-EMF term (tau_stall/free_speed) IS the velocity
        # drag, grounded to the sysid terminal speed (27 rad/s). The MuJoCo joint
        # damping must be ~0 to avoid double-counting (it gave a 9.7 rad/s arm).
        self._nom_shoulder_damp = 0.0
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
        # DR strength follows the curriculum (less randomization early, full late).
        dr_scale = self._curriculum_progress if self._curriculum else 1.0
        p = dr_scale if self._domain_rand else 0.0

        def jitter(nom, frac):
            if p == 0 or rng is None:
                return nom
            return nom * rng.uniform(1 - frac, 1 + frac)

        self._max_arm_speed = jitter(self._nom_max_arm_speed, 0.15)   # narrowed: motor is sysid-measured
        self._min_frac = float(np.clip(jitter(self._nom_min_frac, 0.2), 0.1, 0.4))
        self._vel_kv = jitter(self._nom_vel_kv, 0.4)
        self._tau_max = jitter(self._nom_tau_max, 0.3)
        if self._velocity_control:
            self.model.actuator_gainprm[0, 0] = self._vel_kv
            self.model.actuator_biasprm[0, 2] = -self._vel_kv
            self.model.actuator_forcerange[0] = [-self._tau_max, self._tau_max]
        self._free_speed_max = jitter(self._nom_free_speed_max, 0.20)   # narrowed (sysid-measured)
        # Wide stall-torque DR incl. the weak end -- real authority is uncertain
        # and the hardware looked under-actuated.
        self._tau_stall = self._nom_tau_stall if p == 0 else float(rng.uniform(0.045, 0.070))   # narrowed (sysid)
        self._deadband = self._nom_deadband if p == 0 else float(rng.uniform(0.02, 0.08))
        self._filter_alpha = self._nom_filter_alpha if p == 0 else float(rng.uniform(0.35, 0.7))

        # Action latency (control-step delays) + sensor noise scale.
        delay = 0 if p == 0 else int(rng.integers(0, 3))  # 0..2 steps = 0..20 ms
        self._action_buf = collections.deque([0.0] * (delay + 1), maxlen=delay + 1)
        self._theta_pos_noise = 0.0 if p == 0 else float(rng.uniform(0.0, 0.004))
        self._phi_pos_noise = 0.0 if p == 0 else float(rng.uniform(0.0, 0.02))
        self._phi_vel_noise = 0.0 if p == 0 else float(rng.uniform(0.0, 0.08))

        # Mass / damping randomization. Only a small residual bearing friction
        # on top of the motor's back-EMF drag (which is in the voltage model).
        self.model.dof_damping[0] = 0.0 if p == 0 else float(rng.uniform(0.0, 0.0006))
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
        return np.array(
            [np.cos(theta), np.sin(theta), self._theta_dot_filt, phi, phi_dot, self._last_u],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def set_progress(self, p: float) -> None:
        """Curriculum progress 0 (easy) -> 1 (full difficulty). Called during training."""
        self._curriculum_progress = float(np.clip(p, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_params()
        mujoco.mj_setConst(self.model, self.data)

        rng = self.np_random
        # Start NEAR upright with some velocity -- the catch scenario. The
        # curriculum ramps the spread from near-rest (easy) to full (hard).
        prog = self._curriculum_progress if self._curriculum else 1.0
        opts = options or {}
        theta0, theta_dot0, phi0 = self._sample_start_state(rng, prog, opts)
        self.data.qpos[1] = theta0   # elbow ~ upright
        self.data.qvel[1] = theta_dot0
        self.data.qpos[0] = phi0     # arm
        self.data.qvel[0] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._prev_theta = float(self.data.qpos[1])
        self._theta_dot_filt = float(self.data.qvel[1])
        self._last_u = 0.0
        self._step_count = 0
        self._balance_mode = False
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _sample_start_state(self, rng, prog, opts):
        """Return (theta, theta_dot, phi) for a reset.

        - Explicit override via options {"theta","theta_dot","phi"} (RoA sweep).
        - Corridor mode: angle in the handoff window, then a pendulum speed whose
          mechanical energy is within switch_dE*E_max of upright. This reproduces
          the deployment energy gate, coupling theta and theta_dot the way real
          swing-up handoffs do (no uncatchable super-energetic states).
        - Legacy mode: independent theta x theta_dot rectangle.
        """
        if "theta" in opts or "theta_dot" in opts:
            theta0 = float(opts.get("theta", 0.0))
            theta_dot0 = float(opts.get("theta_dot", 0.0))
            phi0 = float(opts.get("phi", rng.uniform(-np.deg2rad(60.0), np.deg2rad(60.0))))
            return theta0, theta_dot0, phi0

        phi0 = float(rng.uniform(-np.deg2rad(60.0), np.deg2rad(60.0)))
        if self._corridor_reset:
            angle_window = self._start_angle_floor + prog * (self._start_angle - self._start_angle_floor)
            dE_frac = self._switch_dE * (self._dE_floor_frac + prog * (1.0 - self._dE_floor_frac))
            dE_tol = dE_frac * self._E_max
            # Rejection-sample (theta, energy): a rod with energy below the
            # potential at theta cannot physically be there (kinetic < 0). Resample
            # such draws rather than clamping speed to 0, which would pile up fake
            # zero-velocity states at the wide-angle edge of the window.
            theta0, kinetic = 0.0, 0.0
            for _ in range(32):
                theta0 = float(rng.uniform(-angle_window, angle_window))
                energy = self._E_ref + float(rng.uniform(-dE_tol, dE_tol))
                kinetic = energy - self._m_g_lcm * np.cos(theta0)
                if kinetic >= 0.0:
                    break
            else:
                # Fallback (essentially never hit): a guaranteed-valid in-band state.
                theta0, kinetic = 0.0, float(rng.uniform(0.0, dE_tol))
            speed = float(np.sqrt(2.0 * kinetic / self._I_rod))
            speed = min(speed, self._start_vel)   # respect the velocity ceiling
            theta_dot0 = speed * (1.0 if rng.random() < 0.5 else -1.0)
            return theta0, theta_dot0, phi0

        start_angle = self._start_angle_floor + prog * (self._start_angle - self._start_angle_floor)
        start_vel = self._start_vel_floor + prog * (self._start_vel - self._start_vel_floor)
        theta0 = float(rng.uniform(-start_angle, start_angle))
        theta_dot0 = float(rng.uniform(-start_vel, start_vel))
        return theta0, theta_dot0, phi0

    # ------------------------------------------------------------------
    def step(self, action):
        u = float(np.clip(action[0], -self._action_limit, self._action_limit))
        prev_u = self._last_u   # action commanded on the previous step (for smoothness)
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
            phi_dot = float(self.data.qvel[0])
            if self._velocity_control:
                self.data.qfrc_applied[0] = 0.0
            else:
                # DC-motor / voltage model: torque high when slow, dropping to
                # zero at the free-run speed (back-EMF). Matches the grab test
                # and the under-actuation seen on hardware.
                tau_motor = self._tau_stall * (drive - phi_dot / self._free_speed_max)
                tau_motor = float(np.clip(tau_motor, -self._tau_stall, self._tau_stall))
                self.data.qfrc_applied[0] = tau_motor
            mujoco.mj_step(self.model, self.data)

        self._last_u = u   # expose the latest action on the next observation
        obs = self._get_obs()
        theta = float(self.data.qpos[1])
        theta = (theta + np.pi) % (2 * np.pi) - np.pi   # wrap to [-pi, pi]
        phi = float(self.data.qpos[0])
        phi_dot = float(self.data.qvel[0])
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
        # Penalize a fast-spinning arm only once near upright (don't choke the catch),
        # and penalize abrupt action changes to discourage chattering on hardware.
        phi_dot_cost = -self._phi_dot_penalty_gain * phi_dot ** 2 * theta_dot_window
        action_smooth_cost = -self._action_smooth_gain * (u - prev_u) ** 2
        upright_hold = angle_error < self._upright_hold_angle and abs(theta_dot) < self._upright_hold_vel
        tight_hold = angle_error < self._tight_hold_angle and abs(theta_dot) < self._tight_hold_vel
        hold_reward = (3.0 if upright_hold else 0.0) + (5.0 if tight_hold else 0.0)

        terminated = False
        if angle_error <= np.deg2rad(10.0):
            self._balance_mode = True
        elif self._balance_mode and angle_error > self._fall_threshold_rad:
            terminated = True

        reward = float(balance + ctrl_cost + penalty + upright_reward + theta_dot_cost
                       + phi_dot_cost + action_smooth_cost + hold_reward)
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
