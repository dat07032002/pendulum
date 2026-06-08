import collections
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

XML_PATH = Path(__file__).with_name("furuta_pendulum.xml")


class FurutaPendulumEnv(gym.Env):
    """
    Gymnasium environment for the Furuta (rotary inverted) pendulum.

    Observation  : [cos(θ), sin(θ), θ_dot, φ, φ_dot]
                   θ = elbow/pendulum angle  (0 = upright, ±π = hanging)
                   φ = shoulder/arm angle    (limited to ±135° = ±2.356 rad)
    Action       : scalar in [-1, 1] → shoulder motor torque in [-0.132, 0.132] N m
    Reward       : cos(θ) − 0.1·u² − 0.5·max(0, |φ|−2.094)²
                   balance term + control cost + soft joint penalty
    Episode      : fixed length, no early termination
    Reset        : pendulum hanging down (θ ≈ -π) + small uniform noise
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, frame_skip: int = 5,
                 domain_rand: bool = True, dr_schedule=None,
                 episode_seconds: float = 10.0,
                 dr_profile: str = "all",
                 elbow_kick_deg: float = 0.0,
                 elbow_kick_count: int = 1,
                 fall_threshold_deg: float = 20.0):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip   # control period (s)
        self.episode_seconds = float(episode_seconds)
        self.max_steps = round(self.episode_seconds / self.dt)  # steps per episode

        # θ_dot: max free-fall from upright = 36.6 rad/s, 1.5x margin → 55
        # φ_dot: terminal speed = τ_max/damping = 0.132/0.01 = 13.2 rad/s, 1.5x → 20
        # φ:     hard joint limit ±135° = ±2.356 rad; use 2.5 for small margin
        obs_high = np.array([1.0, 1.0, 55.0, 2.5, 20.0], dtype=np.float32)

        self._phi_soft_limit = 2.094   # ±120° — penalty activates beyond this
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._renderer = None   # off-screen renderer (rgb_array)
        self._viewer = None     # interactive viewer (human)
        self._step_count = 0

        # domain randomization
        self._domain_rand          = domain_rand
        self._dr_schedule          = dr_schedule   # shared dict {"progress": 0.0..1.0}
        self._dr_profile           = dr_profile
        self._nom_arm_mass         = float(self.model.body_mass[2])
        self._nom_rod_mass         = float(self.model.body_mass[3])
        self._nom_arm_inertia      = self.model.body_inertia[2].copy()
        self._nom_rod_inertia      = self.model.body_inertia[3].copy()
        self._nom_shoulder_damping = float(self.model.dof_damping[0])
        self._nom_elbow_damping    = float(self.model.dof_damping[1])
        self._nom_motor_gear       = float(self.model.actuator_gear[0, 0])
        self._mass_dr_range        = 0.05
        self._shoulder_damp_dr_range = 0.05
        self._elbow_damp_dr_max    = 0.003
        self._real_ready_elbow_damp_dr_max = 0.00002
        self._motor_gear_dr_range  = 0.05
        self._max_delay            = 1
        self._shoulder_zero_offset_range = np.deg2rad(1.0)
        self._elbow_zero_offset_range    = np.deg2rad(0.5)
        self._shoulder_zero_offset = 0.0
        self._elbow_zero_offset    = 0.0
        # elbow velocity filter state (AS5600 must differentiate)
        self._prev_elbow_angle     = 0.0
        self._elbow_dot_filtered   = 0.0
        self._filter_alpha         = 0.7
        self._action_buf           = collections.deque([0.0], maxlen=1)
        self._balance_mode         = False
        self._balance_steps        = 0
        self._elbow_kick_rad       = np.deg2rad(float(elbow_kick_deg))
        self._elbow_kick_count     = max(0, int(elbow_kick_count))
        self._fall_threshold_rad   = np.deg2rad(float(fall_threshold_deg))
        self._elbow_kicks_applied  = 0
        self._next_elbow_kick_step = 0

        valid_profiles = {
            "all", "mass", "shoulder_damping", "elbow_damping",
            "motor", "delay", "sensor", "sensor_pos", "sensor_vel",
            "sensor_filter", "sensor_elbow_pos_small",
            "sensor_filter_narrow", "mass_motor_delay",
            "real_ready", "real_ready_stage2", "none",
        }
        if self._dr_profile not in valid_profiles:
            raise ValueError(f"Unknown dr_profile {self._dr_profile!r}; expected one of {sorted(valid_profiles)}")

    def _dr_enabled(self, name: str) -> bool:
        if self._dr_profile == "mass_motor_delay":
            return name in {"mass", "motor", "delay"}
        if self._dr_profile in {"real_ready", "real_ready_stage2"}:
            return name in {"mass", "shoulder_damping", "elbow_damping", "motor", "delay"}
        return self._dr_profile == "all" or self._dr_profile == name

    def _sensor_profile(self, name: str) -> bool:
        if self._dr_profile == "all" or self._dr_profile == "sensor":
            return True
        if self._dr_profile == "sensor_pos":
            return name in {"shoulder_pos", "elbow_pos"}
        if self._dr_profile == "sensor_vel":
            return name == "shoulder_vel"
        if self._dr_profile == "sensor_filter":
            return name == "elbow_filter"
        if self._dr_profile == "sensor_elbow_pos_small":
            return name == "elbow_pos_small"
        if self._dr_profile == "sensor_filter_narrow":
            return name == "elbow_filter_narrow"
        if self._dr_profile in {"real_ready", "real_ready_stage2"}:
            return name in {"shoulder_pos", "shoulder_vel", "elbow_pos_small"}
        return False

    def _zero_offset_enabled(self) -> bool:
        return self._dr_profile == "real_ready_stage2"

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        if self._dr_schedule is not None:
            _p = float(self._dr_schedule["progress"])
        elif self._domain_rand:
            _p = 1.0
        else:
            _p = 0.0

        # --- Shoulder (motor encoder, pre-gearbox) ---
        phi     = float(self.data.qpos[0])
        phi_dot = float(self.data.qvel[0])
        phi += self._shoulder_zero_offset
        if _p > 0 and self._sensor_profile("shoulder_pos"):
            phi     += self.np_random.normal(0, 0.017 * _p)
        if _p > 0 and self._sensor_profile("shoulder_vel"):
            phi_dot += self.np_random.normal(0, 0.05 * _p)

        # --- Elbow (AS5600, direct on pivot) ---
        theta = float(self.data.qpos[1])
        theta += self._elbow_zero_offset
        if _p > 0 and self._sensor_profile("elbow_pos"):
            theta += self.np_random.normal(0, 0.003 * _p)
        elif _p > 0 and self._sensor_profile("elbow_pos_small"):
            theta += self.np_random.normal(0, 0.001 * _p)
        if _p > 0 and (self._sensor_profile("elbow_filter") or self._sensor_profile("elbow_filter_narrow")):
            raw_dot = (theta - self._prev_elbow_angle) / self.dt
            self._elbow_dot_filtered = (self._filter_alpha * raw_dot +
                                        (1 - self._filter_alpha) * self._elbow_dot_filtered)
            self._prev_elbow_angle = theta
            theta_dot = self._elbow_dot_filtered
        else:
            theta_dot = float(self.data.qvel[1])

        return np.array(
            [np.cos(theta), np.sin(theta), theta_dot, phi, phi_dot],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        if self._dr_schedule is not None:
            p = float(self._dr_schedule["progress"])
        elif self._domain_rand:
            p = 1.0
        else:
            p = 0.0

        self.model.body_mass[2] = self._nom_arm_mass
        self.model.body_mass[3] = self._nom_rod_mass
        self.model.body_inertia[2] = self._nom_arm_inertia
        self.model.body_inertia[3] = self._nom_rod_inertia
        self.model.dof_damping[0] = self._nom_shoulder_damping
        self.model.dof_damping[1] = self._nom_elbow_damping
        self.model.actuator_gear[0, 0] = self._nom_motor_gear
        self._shoulder_zero_offset = 0.0
        self._elbow_zero_offset    = 0.0

        if p > 0:
            if self._dr_enabled("mass"):
                arm_mass_scale = self.np_random.uniform(1 - self._mass_dr_range*p, 1 + self._mass_dr_range*p)
                rod_mass_scale = self.np_random.uniform(1 - self._mass_dr_range*p, 1 + self._mass_dr_range*p)
                self.model.body_mass[2] = self._nom_arm_mass * arm_mass_scale
                self.model.body_mass[3] = self._nom_rod_mass * rod_mass_scale
                self.model.body_inertia[2] = self._nom_arm_inertia * arm_mass_scale
                self.model.body_inertia[3] = self._nom_rod_inertia * rod_mass_scale
            if self._dr_enabled("shoulder_damping"):
                self.model.dof_damping[0] = self._nom_shoulder_damping * self.np_random.uniform(
                    1 - self._shoulder_damp_dr_range*p,
                    1 + self._shoulder_damp_dr_range*p,
                )
            if self._dr_enabled("elbow_damping"):
                max_elbow_damping = self._elbow_damp_dr_max
                if self._dr_profile in {"real_ready", "real_ready_stage2"}:
                    max_elbow_damping = self._real_ready_elbow_damp_dr_max
                self.model.dof_damping[1] = self.np_random.uniform(0.0, max_elbow_damping * p)
            if self._dr_enabled("motor"):
                self.model.actuator_gear[0, 0] = self._nom_motor_gear * self.np_random.uniform(
                    1 - self._motor_gear_dr_range*p,
                    1 + self._motor_gear_dr_range*p,
                )
            if self._dr_enabled("delay"):
                delay = int(self.np_random.integers(0, round(p) + 1))
                self._action_buf = collections.deque([0.0] * (delay + 1), maxlen=delay + 1)
            if self._dr_profile == "sensor":
                self._filter_alpha = 0.7 + self.np_random.uniform(-0.1*p, 0.1*p)
            elif self._dr_profile == "sensor_filter":
                self._filter_alpha = 0.7 + self.np_random.uniform(-0.1*p, 0.1*p)
            elif self._dr_profile == "sensor_filter_narrow":
                self._filter_alpha = self.np_random.uniform(0.7, 0.8)
            if self._zero_offset_enabled():
                self._shoulder_zero_offset = self.np_random.uniform(
                    -self._shoulder_zero_offset_range * p,
                    self._shoulder_zero_offset_range * p,
                )
                self._elbow_zero_offset = self.np_random.uniform(
                    -self._elbow_zero_offset_range * p,
                    self._elbow_zero_offset_range * p,
                )
        else:
            self._action_buf   = collections.deque([0.0], maxlen=1)
            self._filter_alpha = 0.7

        mujoco.mj_setConst(self.model, self.data)

        noise = 0.05
        self.data.qpos[0] = self.np_random.uniform(-noise, noise)           # shoulder
        self.data.qpos[1] = -np.pi + self.np_random.uniform(-noise, noise)  # elbow (hanging)
        self.data.qvel[:] = self.np_random.uniform(-noise, noise, 2)
        mujoco.mj_forward(self.model, self.data)

        # initialize elbow filter state from reset position
        self._prev_elbow_angle   = float(self.data.qpos[1])
        self._elbow_dot_filtered = float(self.data.qvel[1])
        self._balance_mode       = False
        self._balance_steps      = 0
        self._elbow_kicks_applied = 0
        self._next_elbow_kick_step = int(self.np_random.integers(100, 1001))

        self._step_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self._action_buf.append(float(np.clip(action[0], -1.0, 1.0)))
        self.data.ctrl[0] = self._action_buf[0]
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if (
            self._elbow_kick_rad > 0
            and self._balance_mode
            and self._elbow_kicks_applied < self._elbow_kick_count
            and self._balance_steps >= self._next_elbow_kick_step
        ):
            self.data.qpos[1] += self.np_random.uniform(-self._elbow_kick_rad, self._elbow_kick_rad)
            self._prev_elbow_angle = float(self.data.qpos[1])
            self._elbow_dot_filtered = float(self.data.qvel[1])
            self._elbow_kicks_applied += 1
            self._next_elbow_kick_step = self._balance_steps + int(self.np_random.integers(100, 1001))
            mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        theta = float(self.data.qpos[1])
        phi   = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[1])
        u         = float(self.data.ctrl[0])
        balance   = np.cos(theta)
        ctrl_cost = -0.1 * u ** 2
        excess    = max(0.0, abs(phi) - self._phi_soft_limit)
        penalty   = -0.5 * excess ** 2

        # Once the pendulum is near upright, reward quiet balance instead of
        # fast pass-throughs that briefly cross the upright position.
        angle_error = abs((theta + np.pi) % (2 * np.pi) - np.pi)
        upright_bonus = 0.0
        theta_dot_cost = 0.0
        terminated = False
        if angle_error <= np.deg2rad(10.0):
            self._balance_mode = True
            self._balance_steps += 1
            upright_bonus += 0.2
            theta_dot_cost = -0.005 * theta_dot ** 2
            if angle_error <= np.deg2rad(5.0):
                upright_bonus += 0.2
        elif self._balance_mode and angle_error > self._fall_threshold_rad:
            terminated = True

        reward    = float(balance + ctrl_cost + penalty + upright_bonus + theta_dot_cost)
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


# ------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("Running gymnasium env checker...")
    env = FurutaPendulumEnv()
    check_env(env, warn=True)
    print("check_env passed.\n")

    print(f"control dt   : {env.dt*1000:.1f} ms  (frame_skip={env.frame_skip})")
    print(f"max_steps    : {env.max_steps}  ({env.max_steps * env.dt:.1f} s)")
    print(f"obs space    : {env.observation_space}")
    print(f"action space : {env.action_space}\n")

    obs, _ = env.reset(seed=0)
    print(f"Reset obs    : {obs}")
    theta_est = np.degrees(np.arctan2(obs[1], obs[0]))
    print(f"  cos(th)={obs[0]:.3f}  sin(th)={obs[1]:.3f}  (th~{theta_est:.1f} deg)")

    total_reward = 0.0
    for _ in range(env.max_steps):
        obs, rew, term, trunc, _ = env.step(env.action_space.sample())
        total_reward += rew
        if term or trunc:
            break

    print(f"Random-policy episode reward: {total_reward:.2f}  (expected ~-1000 for hanging)")
    env.close()
