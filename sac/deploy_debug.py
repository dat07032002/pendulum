"""
Debug a sim-trained balance policy on hardware: lift-to-catch, and log
theta, theta_dot, phi, and the policy action each step so we can see WHY it
falls (wrong direction? too weak? thrashing? obs mismatch?).

  python deploy_debug.py --model-dir runs/sac_sim/<dir> --best

Lift the pendulum upright, let go, let it fall, Ctrl+C. It prints the catch
trajectory + a diagnosis hint (does u oppose the fall?).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from furuta_hw_env import FurutaHardwareEnv

PROJECT_DIR = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--action-limit", type=float, default=0.4)
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    md = Path(args.model_dir)
    if not md.is_absolute():
        md = PROJECT_DIR / md
    env = FurutaHardwareEnv(port=args.port, action_limit=1.0, recenter=False,
                            lift_start=True, lift_handoff_deg=10.0, episode_seconds=10.0)
    venv = DummyVecEnv([lambda: env])
    norm = "best_vec_normalize.pkl" if args.best else "vec_normalize.pkl"
    vn = VecNormalize.load(str(md / norm), venv); vn.training = False; vn.norm_reward = False
    model = SAC.load(str(md / ("best_model" if args.best else "latest_model")), device="cpu")
    print(f"policy {model.num_timesteps} steps. Lift upright to catch; Ctrl+C to stop.")

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            traj = []
            done = False
            while not done:
                nobs = vn.normalize_obs(obs.reshape(1, -1))
                a, _ = model.predict(nobs, deterministic=True)
                u = float(np.clip(a.reshape(-1)[0], -args.action_limit, args.action_limit))
                th = np.degrees(np.arctan2(obs[1], obs[0])); thd = float(obs[2]); phi = np.degrees(obs[3])
                traj.append((th, thd, phi, u))
                obs, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
                done = term or trunc
            t = np.array(traj)
            print(f"\n--- episode {ep}: {len(t)} steps ({len(t)*env.control_dt:.2f}s) "
                  f"{info.get('safety_stop','fell/ended')} ---")
            print(" step   theta   thdot     phi      u")
            for i in range(0, len(t), max(1, len(t)//25)):
                print(f" {i:4d}  {t[i,0]:+6.1f}  {t[i,1]:+6.2f}  {t[i,2]:+6.1f}  {t[i,3]:+.3f}")
            # Diagnosis: for balance, u should OPPOSE theta (push arm to correct).
            th, u = t[:, 0], t[:, 3]
            mask = np.abs(th) > 1.0
            if mask.sum() > 3:
                corr = np.corrcoef(th[mask], u[mask])[0, 1]
                print(f" corr(theta, u) = {corr:+.2f}  "
                      f"(consistent strong sign needed; near 0 = not reacting; "
                      f"flipped vs sim = wrong direction)")
                print(f" |u| mean={np.abs(u).mean():.3f} max={np.abs(u).max():.3f} "
                      f"(near action_limit {args.action_limit} = saturating/under-powered)")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
