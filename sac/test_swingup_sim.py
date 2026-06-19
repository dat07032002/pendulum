"""
Tune the energy swing-up in the grounded sim (no hardware needed).

Runs energy_swingup.EnergySwingUp against the sysid-grounded MuJoCo model,
starting from hanging, and reports the pendulum speed (theta_dot) when it first
reaches the top. Goal: deliver the rod near upright SLOW (|theta_dot| < ~5 rad/s)
so the balance policy can catch it.

  python test_swingup_sim.py            # sweep k_e x k_brake
  python test_swingup_sim.py --k 15 --brake 25   # single run, verbose
"""
from __future__ import annotations
import argparse
import numpy as np
import mujoco
from sim_balance_env import FurutaBalanceSimEnv
from energy_swingup import EnergySwingUp


def run(k_e, k_brake, u_max=0.8, phi_swing=80.0, max_s=8.0, seed=0, verbose=False):
    env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=False, action_limit=1.0)
    env.reset(seed=seed)
    # force hanging, at rest, arm centered
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = np.pi
    env.data.qvel[:] = 0.0
    env.data.qvel[1] = 0.5   # small seed velocity (real world is never at perfect rest)
    mujoco.mj_forward(env.model, env.data)
    env._prev_theta = float(env.data.qpos[1])
    env._theta_dot_filt = 0.5

    su = EnergySwingUp(k_e=k_e, u_max=u_max, phi_limit_deg=phi_swing,
                       k_brake=k_brake, brake_umax=u_max)

    obs = env._get_obs()
    steps = int(max_s / env.dt)
    top_thd = None
    t_reach = None
    max_phi = 0.0
    for i in range(steps):
        u = su(obs)
        obs, _, _, _, _ = env.step(np.array([u], dtype=np.float32))
        theta = float(np.arctan2(obs[1], obs[0]))
        max_phi = max(max_phi, abs(np.degrees(obs[3])))
        if verbose and i % 25 == 0:
            print(f"  t={i*env.dt:4.2f}s theta={np.degrees(theta):+7.1f} "
                  f"thd={obs[2]:+6.2f} phi={np.degrees(obs[3]):+6.1f} u={u:+.2f}")
        if abs(theta) < np.deg2rad(15.0):
            top_thd = abs(float(obs[2]))
            t_reach = i * env.dt
            break
    env.close()
    return top_thd, t_reach, max_phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=None, help="single k_energy")
    ap.add_argument("--brake", type=float, default=None, help="single k_brake")
    args = ap.parse_args()

    if args.k is not None:
        thd, t, mphi = run(args.k, args.brake or 20.0, verbose=True)
        if thd is None:
            print(f"\nk_e={args.k} brake={args.brake}: did NOT reach upright (max|phi|={mphi:.0f}deg)")
        else:
            print(f"\nk_e={args.k} brake={args.brake}: reached top in {t:.2f}s at |thd|={thd:.1f} rad/s")
        return

    print("Sweep: delivery speed |theta_dot| at the top (lower = easier catch). '--' = didn't reach.\n")
    k_es = [5, 8, 12, 18]
    brakes = [10, 20, 30, 45]
    header = "k_e \\ brake |" + "".join(f"{b:>8}" for b in brakes)
    print(header)
    print("-" * len(header))
    for k_e in k_es:
        row = f"   {k_e:>6}  |"
        for b in brakes:
            thd, t, mphi = run(k_e, b)
            row += f"{('  --  ' if thd is None else f'{thd:5.1f}'):>8}"
        print(row)
    print("\n(numbers = rad/s at the top; aim for <5. '--' = over/under-energized, didn't reach 15deg)")


if __name__ == "__main__":
    main()
