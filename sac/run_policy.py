"""
Run a trained SAC policy live for a few episodes (deterministic, no learning).

Handles the firmware era mismatch: policies trained under the OLD mapping
(pwm = |u|*255) can run on the NEW deadband-compensated firmware
(pwm = 60 + |u|*195) via --old-mapping, which converts each action so the
motor receives the same PWM the policy intends.

Manual start: episode begins only after the arm is hand-centered and the
pendulum hangs still (same flow as manual-recenter training).

Example (the 94k-step pre-firmware-change policy on new firmware):
  python run_policy.py --model-dir runs/sac_hw/20260611_142028 --old-mapping --action-limit 0.7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from furuta_hw_env import FurutaHardwareEnv

PROJECT_DIR = Path(__file__).parent

# Firmware mappings: u -> PWM
OLD_SCALE = 255.0                 # old firmware: pwm = |u| * 255
NEW_FLOOR, NEW_SPAN = 60.0, 195.0  # new firmware: pwm = 60 + |u| * 195


def old_to_new_u(u_old: float) -> float:
    """Convert an old-mapping action into the new-firmware command with the same PWM."""
    pwm = abs(u_old) * OLD_SCALE
    if pwm <= NEW_FLOOR:           # old dead zone: no torque intended
        return 0.0
    return float(np.sign(u_old) * (pwm - NEW_FLOOR) / NEW_SPAN)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a trained SAC policy on the hardware.")
    parser.add_argument("--model-dir", required=True, help="Run directory with latest_model.zip + vec_normalize.pkl")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-seconds", type=float, default=10.0)
    parser.add_argument("--action-limit", type=float, default=0.7,
                        help="Clamp in the POLICY's action space (pre-conversion)")
    parser.add_argument("--phi-limit-deg", type=float, default=105.0)
    parser.add_argument("--old-mapping", action="store_true",
                        help="Policy was trained on the old firmware (pwm=|u|*255); convert actions")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_DIR / model_dir

    env = FurutaHardwareEnv(
        port=args.port,
        episode_seconds=args.episode_seconds,
        action_limit=1.0,            # we clamp ourselves, pre-conversion
        phi_limit_deg=args.phi_limit_deg,
        recenter=False,              # manual start, like manual-recenter training
    )
    venv = DummyVecEnv([lambda: env])
    vec_norm = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), venv)
    vec_norm.training = False
    vec_norm.norm_reward = False

    model = SAC.load(str(model_dir / "latest_model"), device="cpu")
    print(f"Loaded policy: {model.num_timesteps} training steps "
          f"({'OLD mapping, converting' if args.old_mapping else 'native mapping'})")
    print(">>> Press Enter to begin...", flush=True)
    try:
        input()
    except EOFError:
        pass

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            total, steps, upright = 0.0, 0, 0
            best_streak, streak = 0, 0
            done = False
            while not done:
                nobs = vec_norm.normalize_obs(obs.reshape(1, -1))
                action, _ = model.predict(nobs, deterministic=True)
                u = float(np.clip(action.reshape(-1)[0], -args.action_limit, args.action_limit))
                if args.old_mapping:
                    u = old_to_new_u(u)
                obs, rew, term, trunc, info = env.step(np.array([u], dtype=np.float32))
                total += rew
                steps += 1
                if rew > 1.0:
                    upright += 1
                    streak += 1
                    best_streak = max(best_streak, streak)
                else:
                    streak = 0
                done = term or trunc
            print(f"episode {ep}: reward={total:+.1f}  steps={steps}  "
                  f"upright_steps={upright}  best_hold={best_streak*0.02:.2f}s"
                  + (f"  [{info.get('safety_stop')}]" if info.get("safety_stop") else ""))
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
