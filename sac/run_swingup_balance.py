"""
Swing-up + balance controller for the Furuta pendulum.

  Swing-up  : Åström energy pump (no training needed — pure physics).
  Balance   : Trained SAC policy loaded from a sim training run.
  Switching : |theta| < SWITCH_IN  -> balance
              |theta| > SWITCH_OUT -> back to swing-up

Usage:
  python sac/run_swingup_balance.py \\
      --model-dir runs/sac_sim/20260618_135955 --best --port COM5

Tuning flags (start with defaults, adjust on hardware):
  --k-energy        energy-pump gain      (default 15 — raise if won't reach upright)
  --swingup-umax    swing-up arm ceiling  (default 0.8)
  --phi-swing-deg   arm travel during swing-up (default 80 deg)
  --switch-in-deg   enter balance angle   (default 20 deg)
  --switch-out-deg  exit balance angle    (default 35 deg)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import serial
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from energy_swingup import EnergySwingUp
from sim_balance_env import FurutaBalanceSimEnv

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
CONTROL_HZ = 100
CONTROL_DT  = 1.0 / CONTROL_HZ


def parse_obs(line: str) -> np.ndarray | None:
    m = OBS_RE.search(line)
    if not m:
        return None
    try:
        vals = [float(x) for x in m.group(1).split(",")]
        return np.array(vals, dtype=np.float32) if len(vals) == 5 else None
    except ValueError:
        return None


class SerialReader:
    """Background thread that continuously reads serial lines so the main
    control loop never blocks waiting for data."""

    def __init__(self, ser: serial.Serial):
        import threading
        self._ser = ser
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
                obs = parse_obs(line)
                if obs is not None:
                    with self._lock:
                        self._latest = obs
            except Exception:
                break

    def get(self) -> np.ndarray | None:
        with self._lock:
            obs, self._latest = self._latest, None
        return obs


def load_balance_policy(model_dir: Path, best: bool):
    model_name = "best_model"    if best else "latest_model"
    norm_name  = "best_vec_normalize.pkl" if best else "vec_normalize.pkl"
    dummy_venv = DummyVecEnv([lambda: FurutaBalanceSimEnv(domain_rand=False)])
    vec_norm   = VecNormalize.load(str(model_dir / norm_name), dummy_venv)
    vec_norm.training   = False
    vec_norm.norm_reward = False
    model = SAC.load(str(model_dir / model_name), device="cpu")
    return model, vec_norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Swing-up + balance for Furuta pendulum.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--k-energy",       type=float, default=2.5,
                        help="Energy-pump gain (2.5 at MAX_SPEED=0.40; raise if won't reach upright)")
    parser.add_argument("--swingup-umax",   type=float, default=0.8,
                        help="Arm command ceiling during swing-up [0..1]")
    parser.add_argument("--phi-swing-deg",  type=float, default=80.0,
                        help="Arm travel limit during swing-up [deg]")
    parser.add_argument("--balance-limit",  type=float, default=1.0,
                        help="SAC action clamp for balance [0..1]")
    parser.add_argument("--switch-in-deg",  type=float, default=20.0,
                        help="Enter balance when |theta| < this [deg]")
    parser.add_argument("--switch-out-deg", type=float, default=35.0,
                        help="Exit balance when |theta| > this [deg]")
    parser.add_argument("--switch-vel", type=float, default=9.0,
                        help="Hand off only when |theta_dot| < this [rad/s]. Set BELOW the typical "
                             "delivery so the swing-up retries until it hands off a catchable (slow) rod.")
    parser.add_argument("--coast",          type=float, default=0.15,
                        help="Legacy coast fraction (ignored when --brake > 0)")
    parser.add_argument("--brake",          type=float, default=7.0,
                        help="Asymmetric brake gain when dE<0 (excess energy). Tuned to 7 with k_energy=40.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / model_dir

    print(f"Loading balance policy: {model_dir} ({'best' if args.best else 'latest'})...")
    model, vec_norm = load_balance_policy(model_dir, args.best)
    print(f"  Loaded ({model.num_timesteps:,} training steps)")

    swingup = EnergySwingUp(
        k_e=args.k_energy,
        u_max=args.swingup_umax,
        phi_limit_deg=args.phi_swing_deg,
        coast_fraction=args.coast,
        k_brake=args.brake,
        brake_umax=0.8,
    )

    switch_in  = np.deg2rad(args.switch_in_deg)
    switch_out = np.deg2rad(args.switch_out_deg)

    print(f"\nSwitch IN  (-> balance) : |theta| < {args.switch_in_deg:.0f} deg")
    print(f"Switch OUT (-> swing-up): |theta| > {args.switch_out_deg:.0f} deg")
    print(f"Connecting to {args.port} at 921600 baud...")

    with serial.Serial(args.port, 921600, timeout=0.05) as ser:
        time.sleep(2.0)
        ser.write(b"z\n")   # zero arm encoder at start
        time.sleep(0.2)
        reader = SerialReader(ser)
        time.sleep(0.1)  # let reader thread buffer a few obs

        print("\nArm encoder zeroed. Starting swing-up (pendulum should be hanging).")
        print("Ctrl+C to stop.\n")

        mode          = "swingup"
        balance_start = 0.0
        best_hold     = 0.0
        balance_count = 0
        obs_miss      = 0
        last_obs      = None

        try:
            while True:
                t0 = time.perf_counter()

                obs = reader.get()
                if obs is None:
                    obs_miss += 1
                    if obs_miss % 20 == 0:
                        print(f"  WARNING: {obs_miss} consecutive obs timeouts")
                    if last_obs is not None and obs_miss < 5:
                        obs = last_obs  # use stale obs briefly
                    else:
                        ser.write(b"u 0.0\n")
                        time.sleep(CONTROL_DT)
                        continue
                else:
                    obs_miss = 0
                    last_obs = obs

                cos_th, sin_th, th_dot, phi, phi_dot = obs
                theta    = float(np.arctan2(sin_th, cos_th))
                abs_theta = abs(theta)

                # ---- State machine ----
                if mode == "swingup" and abs_theta < switch_in and abs(th_dot) < args.switch_vel:
                    mode = "balance"
                    balance_start = time.time()
                    balance_count += 1
                    print(f"  [{balance_count}] -> BALANCE  "
                          f"theta={np.degrees(theta):+5.1f}deg  "
                          f"th_dot={th_dot:+5.1f} rad/s")

                elif mode == "balance" and abs_theta > switch_out:
                    hold = time.time() - balance_start
                    best_hold = max(best_hold, hold)
                    print(f"  [{balance_count}] -> SWING-UP  "
                          f"held={hold:.2f}s  best={best_hold:.2f}s")
                    mode = "swingup"

                # ---- Control ----
                if mode == "balance":
                    nobs = vec_norm.normalize_obs(obs.reshape(1, -1))
                    action, _ = model.predict(nobs, deterministic=True)
                    u = float(np.clip(action.flat[0], -args.balance_limit, args.balance_limit))
                else:
                    u = swingup(obs)

                ser.write(f"u {u:.4f}\n".encode())

                # 100 Hz pacing
                elapsed = time.perf_counter() - t0
                wait    = CONTROL_DT - elapsed
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            ser.write(b"u 0.0\n")
            time.sleep(0.05)
            ser.write(b"s\n")

    print(f"\nDone. Balance attempts: {balance_count}  Best hold: {best_hold:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
