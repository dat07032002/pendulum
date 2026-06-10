"""
Record one episode of the trained policy and save as a GIF + behaviour plot.
Usage:  python record.py [model_path] [output.gif]
"""
import sys
import json
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from furuta_env import FurutaPendulumEnv

FPS = 50   # matches env render_fps; 2000 steps / 50 fps = 40s gif


def _latest_run_model() -> str:
    runs_dir = Path(__file__).with_name("runs") / "no_dr"
    runs = sorted(runs_dir.iterdir()) if runs_dir.exists() else []
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}. Pass model path explicitly.")
    return str(runs[-1] / "best_model")


MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else _latest_run_model()
OUT_PATH   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("eval.gif")
model_dir = Path(MODEL_PATH).parent
norm_path  = str(model_dir / "vec_normalize_best.pkl")

episode_seconds = 10.0
config_path = model_dir / "run_config.json"
if config_path.exists():
    episode_seconds = float(
        json.loads(config_path.read_text(encoding="ascii")).get(
            "episode_seconds",
            episode_seconds,
        )
    )

model = PPO.load(MODEL_PATH)

env = DummyVecEnv([
    lambda: FurutaPendulumEnv(
        render_mode="rgb_array",
        domain_rand=False,
        episode_seconds=episode_seconds,
    )
])
env = VecNormalize.load(norm_path, env)
env.training   = False
env.norm_reward = False

obs    = env.reset()
frames = []
ep_reward = 0.0

thetas, phis, actions, times = [], [], [], []
t = 0.0
dt = 1.0 / 100.0  # 10 ms control period

while True:
    frame = env.envs[0].render()
    frames.append(frame)

    # raw obs before normalisation: [cos(theta), sin(theta), theta_dot, phi, phi_dot]
    raw = env.get_original_obs()[0]
    theta = float(np.arctan2(raw[1], raw[0]))
    phi   = float(raw[3])

    action, _ = model.predict(obs, deterministic=True)
    thetas.append(np.degrees(theta))
    phis.append(np.degrees(phi))
    actions.append(float(action[0][0]))
    times.append(t)
    t += dt

    obs, reward, done, info = env.step(action)
    ep_reward += float(reward[0])

    if done[0]:
        break

env.close()

print(f"Episode: {len(frames)} frames ({episode_seconds:.1f}s), total reward: {ep_reward:.1f} / {len(frames):.0f}")
print(f"Saving {OUT_PATH} at {FPS} fps ...")
imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
print(f"Saved -> {OUT_PATH.resolve()}")

# --- behaviour plot ---
plot_path = OUT_PATH.with_suffix(".png")
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

axes[0].plot(times, thetas, color="steelblue")
axes[0].axhline(0, color="gray", lw=0.5, ls="--")
axes[0].axhline(10,  color="orange", lw=0.8, ls=":", label="+/-10 deg")
axes[0].axhline(-10, color="orange", lw=0.8, ls=":")
axes[0].axhline(5,   color="green",  lw=0.8, ls=":", label="+/-5 deg")
axes[0].axhline(-5,  color="green",  lw=0.8, ls=":")
axes[0].set_ylabel("theta - pendulum (deg)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Policy behaviour - pendulum angle, arm angle, motor command")

axes[1].plot(times, phis, color="darkorange")
axes[1].axhline(135,  color="red", lw=0.8, ls="--", label="+/-135 deg limit")
axes[1].axhline(-135, color="red", lw=0.8, ls="--")
axes[1].set_ylabel("phi - arm (deg)")
axes[1].legend(loc="upper right", fontsize=8)

axes[2].plot(times, actions, color="crimson", lw=0.8)
axes[2].axhline(0, color="gray", lw=0.5, ls="--")
axes[2].set_ylabel("u - motor command")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
fig.savefig(str(plot_path), dpi=120)
plt.close(fig)
print(f"Plot  -> {plot_path.resolve()}")
