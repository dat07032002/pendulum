"""
Tune the pump-and-coast energy swing-up in the grounded simulation.

Runs EnergySwingUp from the hanging position and reports pendulum speed when it
first reaches the top. The sweep compares pump gain and coast threshold.

  python test_swingup_sim.py
  python test_swingup_sim.py --k 12 --coast 0.15
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from energy_swingup import EnergySwingUp
from sim_balance_env import FurutaBalanceSimEnv


def run(k_e, coast, u_max=0.8, phi_swing=80.0, max_s=8.0, seed=0, verbose=False):
    env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=False, action_limit=1.0)
    env.reset(seed=seed)
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = np.pi
    env.data.qvel[:] = 0.0
    env.data.qvel[1] = 0.5
    mujoco.mj_forward(env.model, env.data)
    env._prev_theta = float(env.data.qpos[1])
    env._theta_dot_filt = 0.5

    swingup = EnergySwingUp(
        k_e=k_e,
        u_max=u_max,
        phi_limit_deg=phi_swing,
        coast_fraction=coast,
    )

    obs = env._get_obs()
    top_thd = None
    t_reach = None
    max_phi = 0.0
    for i in range(int(max_s / env.dt)):
        u = swingup(obs)
        obs, _, _, _, _ = env.step(np.array([u], dtype=np.float32))
        theta = float(np.arctan2(obs[1], obs[0]))
        max_phi = max(max_phi, abs(np.degrees(obs[3])))
        if verbose and i % 25 == 0:
            print(f"  t={i * env.dt:4.2f}s theta={np.degrees(theta):+7.1f} "
                  f"thd={obs[2]:+6.2f} phi={np.degrees(obs[3]):+6.1f} u={u:+.2f}")
        if abs(theta) < np.deg2rad(15.0):
            top_thd = abs(float(obs[2]))
            t_reach = i * env.dt
            break
    env.close()
    return top_thd, t_reach, max_phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=None, help="Single energy-pump gain")
    ap.add_argument("--coast", type=float, default=0.15, help="Coast fraction for a single run")
    args = ap.parse_args()

    if args.k is not None:
        thd, t, mphi = run(args.k, args.coast, verbose=True)
        if thd is None:
            print(f"\nk_e={args.k} coast={args.coast}: did not reach upright "
                  f"(max|phi|={mphi:.0f}deg)")
        else:
            print(f"\nk_e={args.k} coast={args.coast}: reached top in {t:.2f}s "
                  f"at |thd|={thd:.1f} rad/s")
        return

    print("Sweep: delivery speed |theta_dot| at the top (lower is easier to catch). '--' = did not reach.\n")
    k_es = [5, 8, 12, 18]
    coasts = [0.05, 0.10, 0.15, 0.20]
    header = "k_e \\ coast |" + "".join(f"{coast:>8.2f}" for coast in coasts)
    print(header)
    print("-" * len(header))
    for k_e in k_es:
        row = f"   {k_e:>6}  |"
        for coast in coasts:
            thd, _, _ = run(k_e, coast)
            row += f"{('  --  ' if thd is None else f'{thd:5.1f}'):>8}"
        print(row)
    print("\n(numbers are rad/s at the top; aim for a slow, catchable delivery)")


if __name__ == "__main__":
    main()
