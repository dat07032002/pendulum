"""Render a trained balance policy to a GIF (offscreen MuJoCo)."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sim_balance_env import FurutaBalanceSimEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--start-vel", type=float, default=4.0)
    ap.add_argument("--start-angle", type=float, default=20.0)
    ap.add_argument("--out", default="balance.gif")
    ap.add_argument("--fps", type=int, default=33)
    args = ap.parse_args()

    md = Path(args.model_dir)
    if not md.is_absolute():
        md = Path(__file__).parent / md
    norm = "best_vec_normalize.pkl" if args.best else "vec_normalize.pkl"
    mdl = "best_model" if args.best else "latest_model"
    venv = DummyVecEnv([lambda: FurutaBalanceSimEnv(domain_rand=False)])
    vn = VecNormalize.load(str(md / norm), venv); vn.training = False; vn.norm_reward = False
    m = SAC.load(str(md / mdl), device="cpu")

    env = FurutaBalanceSimEnv(domain_rand=False, render_mode="rgb_array",
                              velocity_control=False,   # voltage model (matches --torque training)
                              start_vel=args.start_vel, start_angle_deg=args.start_angle,
                              episode_seconds=6.0)
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            a, _ = m.predict(vn.normalize_obs(obs.reshape(1, -1)), deterministic=True)
            obs, r, term, trunc, _ = env.step(a.reshape(-1))
            frames.append(env.render())
            done = term or trunc
    env.close()

    import imageio.v2 as imageio
    out = Path(__file__).parent / args.out
    imageio.mimsave(str(out), frames, fps=args.fps)
    print(f"saved {out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
