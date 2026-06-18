import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from furuta_env import FurutaPendulumEnv

model_path = sorted((Path(__file__).with_name("runs") / "no_dr").iterdir())[-1] / "best_model"
norm_path = model_path.parent / "vec_normalize_best.pkl"
model = PPO.load(str(model_path), device="cpu")
env = DummyVecEnv([lambda: FurutaPendulumEnv(domain_rand=False, episode_seconds=30.0)])
env = VecNormalize.load(str(norm_path), env)
env.training = False
env.norm_reward = False

obs = env.reset()
thetas, upright_steps = [], 0
for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    theta = float(env.envs[0].data.qpos[1])
    thetas.append(np.degrees(theta))
    if abs(theta) < np.radians(10):
        upright_steps += 1
    if done[0]:
        break

print(f"Steps: {len(thetas)}")
print(f"Upright (<10deg): {upright_steps}/{len(thetas)} = {100*upright_steps/len(thetas):.1f}%")
print(f"theta min={min(thetas):.1f} max={max(thetas):.1f} final={thetas[-1]:.1f} deg")
