"""
Load the best trained PPO policy and run one episode with the interactive viewer.
Usage:
    python eval.py                                    # loads latest no_dr run
    python eval.py runs/no_dr/<timestamp>/best_model  # specific run
    python eval.py models/scratch_dr/best_model       # DR model
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from furuta_env import FurutaPendulumEnv


def _latest_run_model() -> str:
    runs_dir = Path(__file__).with_name("runs") / "no_dr"
    runs = sorted(runs_dir.iterdir()) if runs_dir.exists() else []
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}. Pass model path explicitly.")
    return str(runs[-1] / "best_model")


if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = _latest_run_model()

norm_path = str(Path(model_path).parent / "vec_normalize_best.pkl")

print(f"Loading model : {model_path}")
print(f"Loading stats : {norm_path}")

model = PPO.load(model_path)

env = DummyVecEnv([lambda: FurutaPendulumEnv(render_mode="human")])
env = VecNormalize.load(norm_path, env)
env.training = False
env.norm_reward = False

obs = env.reset()
ep_reward = 0.0
step = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    ep_reward += float(reward[0])
    step += 1
    if done[0]:
        break

print(f"Episode finished — {step} steps, total reward: {ep_reward:.2f}  "
      f"(max possible: {step:.0f})")
env.close()
