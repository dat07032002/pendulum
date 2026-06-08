"""
Record one episode of the trained policy and save as a GIF.
Usage:  python record.py [output.gif]
"""
import sys
import json
import numpy as np
import imageio
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

while True:
    # render before step so first frame is the reset state
    frame = env.envs[0].render()
    frames.append(frame)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    ep_reward += float(reward[0])

    if done[0]:
        break

env.close()

print(f"Episode: {len(frames)} frames ({episode_seconds:.1f}s), total reward: {ep_reward:.1f} / {len(frames):.0f}")
print(f"Saving {OUT_PATH} at {FPS} fps ...")

imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
print(f"Saved -> {OUT_PATH.resolve()}")
