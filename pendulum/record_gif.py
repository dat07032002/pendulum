"""
Record a GIF of the trained policy running in simulation.
Usage:
    python record_gif.py                                          # latest no_dr run
    python record_gif.py runs/no_dr/<timestamp>/best_model       # specific model
    python record_gif.py --out policy.gif --fps 30 --seconds 15  # options
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from furuta_env import FurutaPendulumEnv

try:
    import imageio
except ImportError:
    print("Missing: pip install imageio")
    sys.exit(1)


def latest_no_dr_model() -> Path:
    runs_dir = Path(__file__).with_name("runs") / "no_dr"
    runs = sorted(runs_dir.iterdir()) if runs_dir.exists() else []
    if not runs:
        raise FileNotFoundError(f"No runs in {runs_dir}")
    return runs[-1] / "best_model"


parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default=None)
parser.add_argument("--out", default="policy.gif")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--dr-model", action="store_true", help="Model is from a DR run (no vec_normalize in no_dr/)")
args = parser.parse_args()

model_path = Path(args.model) if args.model else latest_no_dr_model()
norm_path = model_path.parent / "vec_normalize_best.pkl"
config_path = model_path.parent / "run_config.json"

episode_seconds = args.seconds
fixed_motor_deadband = 0.0
action_limit = 1.0
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="ascii"))
    fixed_motor_deadband = float(config.get("fixed_motor_deadband", 0.0))
    action_limit = float(config.get("action_limit", 1.0))

print(f"Model : {model_path}")
print(f"Norm  : {norm_path}")
print(f"Fixed deadband: {fixed_motor_deadband:.3f}")
print(f"Action limit: {action_limit:.3f}")

model = PPO.load(str(model_path))
env = DummyVecEnv([
    lambda: FurutaPendulumEnv(
        render_mode="rgb_array",
        domain_rand=False,
        episode_seconds=episode_seconds,
        fixed_motor_deadband=fixed_motor_deadband,
        action_limit=action_limit,
    )
])
env = VecNormalize.load(str(norm_path), env)
env.training = False
env.norm_reward = False

obs = env.reset()
frames = []
step = 0
max_steps = int(args.seconds * 100)  # 100Hz control

print(f"Recording {args.seconds:.0f}s ({max_steps} steps) ...")
while step < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame = env.envs[0].render()
    if frame is not None:
        frames.append(frame)
    step += 1
    if done[0]:
        obs = env.reset()

env.close()

out_path = Path(args.out)
print(f"Saving {len(frames)} frames -> {out_path} at {args.fps} fps ...")
save_kwargs = {"fps": args.fps}
if out_path.suffix.lower() == ".gif":
    save_kwargs["loop"] = 0
imageio.mimsave(str(out_path), frames, **save_kwargs)
print(f"Done: {out_path} ({out_path.stat().st_size // 1024} KB)")
