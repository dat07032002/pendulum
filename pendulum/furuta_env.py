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
    Action       : scalar in [-1, 1] → shoulder motor torque
    Reward       : cos(θ) − 0.5·max(0, |φ|−2.094)²
                   balance term + soft penalty when arm exceeds ±120°
    Episode      : 10 s fixed length, no early termination
    Reset        : pendulum hanging down (θ ≈ -π) + small uniform noise
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, frame_skip: int = 5):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip   # control period (s)
        self.max_steps = round(10.0 / self.dt)           # steps per episode

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

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        theta = float(self.data.qpos[1])
        phi = float(self.data.qpos[0])
        theta_dot = float(self.data.qvel[1])
        phi_dot = float(self.data.qvel[0])
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

        noise = 0.05
        self.data.qpos[0] = self.np_random.uniform(-noise, noise)           # shoulder
        self.data.qpos[1] = -np.pi + self.np_random.uniform(-noise, noise)  # elbow (hanging)
        self.data.qvel[:] = self.np_random.uniform(-noise, noise, 2)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.data.ctrl[0] = float(np.clip(action[0], -1.0, 1.0))
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        theta = float(self.data.qpos[1])
        phi   = float(self.data.qpos[0])
        balance  = np.cos(theta)
        excess   = max(0.0, abs(phi) - self._phi_soft_limit)
        penalty  = -0.5 * excess ** 2
        reward   = float(balance + penalty)
        self._step_count += 1
        terminated = False
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
