"""
Region-of-attraction sweep for a trained SAC balance policy.

Rolls the deterministic policy from a grid of (theta, theta_dot) start states and
records how long each one stays upright. This is the honest catch metric: it shows
*which* arrival states the policy can hold, not an average over a rectangle that
includes physically uncatchable (super-energetic) corners.

The energy-gated handoff corridor (|E - E_upright| < switch_dE * E_max) is overlaid
so you can see whether the policy covers the states the deployment runner will hand
it. Use the printed safe set as the deployment region-of-attraction gate.

  python roa_sweep.py --model-dir runs/sac_sim/<run> --best
  python roa_sweep.py --model-dir runs/sac_sim/<run> --best --hold-seconds 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from energy_swingup import EnergySwingUp
from sim_balance_env import FurutaBalanceSimEnv


def load_policy(model_dir: Path, best: bool, velocity_control: bool):
    model_name = "best_model" if best else "latest_model"
    norm_name = "best_vec_normalize.pkl" if best else "vec_normalize.pkl"
    venv = DummyVecEnv([lambda: FurutaBalanceSimEnv(domain_rand=False,
                                                    velocity_control=velocity_control)])
    vec_norm = VecNormalize.load(str(model_dir / norm_name), venv)
    vec_norm.training = False
    vec_norm.norm_reward = False
    model = SAC.load(str(model_dir / model_name), device="cpu")
    return model, vec_norm


def hold_duration(env, model, vec_norm, theta, theta_dot) -> float:
    """Longest upright-hold (|theta|<10deg, |theta_dot|<3) from a start state [s]."""
    obs, _ = env.reset(seed=0, options={"theta": theta, "theta_dot": theta_dot, "phi": 0.0})
    streak, best, done = 0, 0, False
    while not done:
        action, _ = model.predict(vec_norm.normalize_obs(obs.reshape(1, -1)), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action.reshape(-1))
        th = float(np.arctan2(obs[1], obs[0]))
        if abs(th) < np.deg2rad(10.0) and abs(obs[2]) < 3.0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
        done = terminated or truncated
    return best * env.dt


def main() -> int:
    ap = argparse.ArgumentParser(description="Region-of-attraction sweep for a balance policy.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--velocity", action="store_true",
                    help="Use the Nidec velocity model (default: torque model, matches --torque training).")
    ap.add_argument("--angle-max", type=float, default=30.0, help="Max |theta| swept [deg]")
    ap.add_argument("--vel-max", type=float, default=12.0, help="Max |theta_dot| swept [rad/s]")
    ap.add_argument("--n-angle", type=int, default=13)
    ap.add_argument("--n-vel", type=int, default=13)
    ap.add_argument("--hold-seconds", type=float, default=2.0, help="Success threshold [s]")
    ap.add_argument("--episode-seconds", type=float, default=8.0)
    ap.add_argument("--switch-dE", type=float, default=0.08, help="Corridor energy tolerance for the overlay.")
    ap.add_argument("--csv", default=None, help="Optional path to write the hold-duration grid as CSV.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / model_dir

    velocity_control = bool(args.velocity)
    model, vec_norm = load_policy(model_dir, args.best, velocity_control)
    env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=velocity_control,
                              episode_seconds=args.episode_seconds, fall_threshold_deg=55.0)

    angles = np.linspace(-args.angle_max, args.angle_max, args.n_angle)        # deg
    vels = np.linspace(args.vel_max, -args.vel_max, args.n_vel)                # rad/s (top row = +)

    # Energy corridor: |0.5*I*thd^2 + m g lcm cos(theta) - E_ref| < switch_dE * E_max.
    swing = EnergySwingUp()
    dE_tol = args.switch_dE * swing.E_max

    grid = np.zeros((args.n_vel, args.n_angle), dtype=np.float32)
    print(f"Policy: {model_dir} ({'best' if args.best else 'latest'}), "
          f"{'velocity' if velocity_control else 'torque'} model")
    print(f"Hold success threshold: {args.hold_seconds:.1f}s  (cell = hold seconds; "
          f"'*' inside energy corridor, '#' = success)\n")

    header = "thd\\th |" + "".join(f"{a:6.0f}" for a in angles)
    print(header)
    print("-" * len(header))
    for i, thd in enumerate(vels):
        row_cells = []
        for j, a_deg in enumerate(angles):
            theta = np.deg2rad(a_deg)
            hold = hold_duration(env, model, vec_norm, theta, float(thd))
            grid[i, j] = hold
            energy = 0.5 * swing.I_ROD * thd ** 2 + swing.M_ROD * swing.G * swing.L_CM * np.cos(theta)
            in_corridor = abs(energy - swing.E_ref) < dE_tol and abs(a_deg) <= args.angle_max
            if hold >= args.hold_seconds:
                mark = "#"
            elif in_corridor:
                mark = "*"
            else:
                mark = " "
            row_cells.append(f"{hold:4.1f}{mark}")
        print(f"{thd:+6.1f} |" + "".join(row_cells))
    env.close()

    # Summary, restricted to the corridor states the policy will actually be handed.
    corridor_mask = np.zeros_like(grid, dtype=bool)
    for i, thd in enumerate(vels):
        for j, a_deg in enumerate(angles):
            energy = (0.5 * swing.I_ROD * thd ** 2
                      + swing.M_ROD * swing.G * swing.L_CM * np.cos(np.deg2rad(a_deg)))
            corridor_mask[i, j] = abs(energy - swing.E_ref) < dE_tol

    success = grid >= args.hold_seconds
    n_corr = int(corridor_mask.sum())
    corr_success = int((success & corridor_mask).sum())
    print(f"\nGrid success  (>= {args.hold_seconds:.1f}s): {int(success.sum())}/{grid.size} cells")
    if n_corr:
        print(f"Corridor success (handoff set): {corr_success}/{n_corr} cells "
              f"({100.0 * corr_success / n_corr:.0f}%)")
    print(f"Median hold over corridor cells: {np.median(grid[corridor_mask]):.2f}s"
          if n_corr else "No corridor cells in the swept range.")

    if args.csv:
        out = Path(args.csv)
        if not out.is_absolute():
            out = Path(__file__).parent / out
        np.savetxt(out, grid, delimiter=",", fmt="%.3f",
                   header="rows=theta_dot(+max..-max), cols=theta(-max..+max); values=hold[s]")
        print(f"Saved hold grid to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
