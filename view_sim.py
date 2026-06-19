"""
View the ACTUAL training physics (sim_balance_env: voltage motor model + cable
spring + grounded 24H404H160 params) in the MuJoCo viewer — to verify the model
and dynamics look right before trusting a trained policy.

Run it in YOUR terminal (needs a display + keyboard):

  python view_sim.py                                  # manual: push arm by hand
  python view_sim.py --model-dir sac/runs/sac_sim/<dir> --best   # watch the policy balance

Keys (type in THIS terminal, not the viewer window):
  A / D   push arm left / right        SPACE  stop arm
  R       reset                        Q      quit
"""
from __future__ import annotations
import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

try:
    import msvcrt
except ImportError:
    msvcrt = None

sys.path.insert(0, str(Path(__file__).parent / "sac"))
from sim_balance_env import FurutaBalanceSimEnv


def _hang(env):
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = np.pi
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env._prev_theta = float(env.data.qpos[1])
    env._theta_dot_filt = 0.0
    return env._get_obs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="run a trained policy from this dir")
    ap.add_argument("--best", action="store_true")
    args = ap.parse_args()

    env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=False, action_limit=1.0)
    env.reset()

    policy = vec_norm = None
    if args.model_dir:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = Path(__file__).parent / md
        venv = DummyVecEnv([lambda: FurutaBalanceSimEnv(domain_rand=False)])
        norm = "best_vec_normalize.pkl" if args.best else "vec_normalize.pkl"
        vec_norm = VecNormalize.load(str(md / norm), venv)
        vec_norm.training = False
        policy = SAC.load(str(md / ("best_model" if args.best else "latest_model")), device="cpu")
        print(f"Loaded policy: {policy.num_timesteps:,} steps — starts near upright, balances.")
        obs, _ = env.reset()
    else:
        obs = _hang(env)   # manual mode: start hanging, push it around

    state = {"u": 0.0, "reset": False, "quit": False}

    def keys():
        while not state["quit"]:
            if msvcrt and msvcrt.kbhit():
                k = msvcrt.getch().lower()
                if k == b'a': state["u"] = -0.5; print("  arm <- (u=-0.5)")
                elif k == b'd': state["u"] = 0.5; print("  arm -> (u=+0.5)")
                elif k == b' ': state["u"] = 0.0; print("  stop")
                elif k == b'r': state["reset"] = True
                elif k == b'q': state["quit"] = True
            time.sleep(0.02)
    threading.Thread(target=keys, daemon=True).start()

    print("=" * 50)
    print("  A/D push arm | SPACE stop | R reset | Q quit")
    print("  (type in THIS terminal, not the viewer)")
    print(f"  motor: tau_stall={env._nom_tau_stall} N·m, free_speed={env._nom_free_speed_max} rad/s")
    print("=" * 50)

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.5, -20, 135
        t = time.perf_counter()
        while v.is_running() and not state["quit"]:
            if state["reset"]:
                obs, _ = env.reset() if policy else (None, None)
                if not policy:
                    obs = _hang(env)
                state["reset"] = False

            if policy is not None:
                nobs = vec_norm.normalize_obs(obs.reshape(1, -1))
                action, _ = policy.predict(nobs, deterministic=True)
                cmd = float(action.flat[0])
            else:
                cmd = state["u"]

            obs, _, term, trunc, _ = env.step(np.array([cmd], dtype=np.float32))
            theta = np.degrees(np.arctan2(obs[1], obs[0]))
            print(f"\r  theta={theta:+6.1f}  th_dot={obs[2]:+6.2f}  phi={np.degrees(obs[3]):+6.1f}  "
                  f"u={cmd:+.2f}   ", end="", flush=True)
            if term or trunc:
                obs, _ = env.reset() if policy else (None, None)
                if not policy:
                    obs = _hang(env)
            v.sync()
            t += env.dt
            s = t - time.perf_counter()
            if s > 0:
                time.sleep(s)

    state["quit"] = True
    env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
