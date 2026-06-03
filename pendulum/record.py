"""
Record one episode of the trained policy and save as a GIF.
Usage:  python record.py [output.gif]
"""
import sys
import numpy as np
import imageio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from furuta_env import FurutaPendulumEnv

MODELS_DIR  = Path(__file__).with_name("models")
OUT_PATH    = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("eval.gif")
FPS         = 50   # matches env render_fps; 1000 steps / 50 fps = 20s gif

model = PPO.load(str(MODELS_DIR / "best_model"))

env = DummyVecEnv([lambda: FurutaPendulumEnv(render_mode="rgb_array")])
env = VecNormalize.load(str(MODELS_DIR / "vec_normalize_best.pkl"), env)
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

print(f"Episode: {len(frames)} frames, total reward: {ep_reward:.1f} / {len(frames):.0f}")
print(f"Saving {OUT_PATH} at {FPS} fps ...")

imageio.mimsave(str(OUT_PATH), frames, fps=FPS, loop=0)
print(f"Saved -> {OUT_PATH.resolve()}")
