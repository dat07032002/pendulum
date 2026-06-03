"""
Load the best trained PPO policy and run one episode with the interactive viewer.
Usage:
    python eval.py                      # loads models/best_model.zip
    python eval.py models/ppo_furuta_final  # load a specific checkpoint (no .zip)
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from furuta_env import FurutaPendulumEnv

MODELS_DIR = Path(__file__).with_name("models")
model_path = sys.argv[1] if len(sys.argv) > 1 else str(MODELS_DIR / "best_model")
norm_path  = str(MODELS_DIR / "vec_normalize_best.pkl")

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
