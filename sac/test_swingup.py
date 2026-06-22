"""
Swing-up only test — no balance policy needed.

Runs the energy pump continuously so you can tune k_e safely
without the balance policy switching in.

Usage:
  python sac/test_swingup.py --port COM6 --k-energy 5

Tuning:
  Too violent  -> lower --k-energy  (try 3, 5, 7)
  Won't reach upright -> raise --k-energy  (try 10, 15, 20)
  Reaches but overshoots -> lower --swingup-umax  (try 0.5, 0.4)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).parent))
from energy_swingup import EnergySwingUp

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
CONTROL_DT = 0.01   # 100 Hz


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
    def __init__(self, ser: serial.Serial):
        import threading
        self._ser = ser
        self._latest = None
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

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

    def get(self):
        with self._lock:
            obs, self._latest = self._latest, None
        return obs


def main() -> int:
    parser = argparse.ArgumentParser(description="Swing-up only test.")
    parser.add_argument("--port",          default="COM5")
    parser.add_argument("--k-energy",      type=float, default=5.0)
    parser.add_argument("--swingup-umax",  type=float, default=0.8)
    parser.add_argument("--phi-swing-deg", type=float, default=80.0)
    parser.add_argument("--coast",         type=float, default=0.15,
                        help="Coast when energy deficit is below this fraction of the swing-up energy range.")
    parser.add_argument("--u-floor",       type=float, default=0.0,
                        help="Optional minimum non-zero swing-up command. Try 0.08 if small commands do nothing.")
    parser.add_argument("--k-center",      type=float, default=0.15,
                        help="Arm-centering gain that subtracts k_center*phi from u.")
    parser.add_argument("--k-arm-damp",    type=float, default=0.0,
                        help="Arm damping gain that subtracts k_arm_damp*phi_dot from u.")
    parser.add_argument("--stop-deg",      type=float, default=10.0,
                        help="Stop motor when |theta| < this angle [deg] (default 10)")
    args = parser.parse_args()

    swingup = EnergySwingUp(
        k_e=args.k_energy,
        u_max=args.swingup_umax,
        phi_limit_deg=args.phi_swing_deg,
        coast_fraction=args.coast,
        k_center=args.k_center,
        k_arm_damp=args.k_arm_damp,
        u_floor=args.u_floor,
    )

    print(f"Connecting to {args.port}  k_energy={args.k_energy}  umax={args.swingup_umax}  "
          f"coast={args.coast}  u_floor={args.u_floor}  k_center={args.k_center}  "
          f"k_arm_damp={args.k_arm_damp}")
    with serial.Serial(args.port, 921600, timeout=0.05) as ser:
        time.sleep(2.0)
        ser.write(b"z\n")
        time.sleep(0.1)
        reader = SerialReader(ser)
        time.sleep(0.1)
        stop_rad = np.deg2rad(args.stop_deg)
        print(f"Running swing-up. Stops when |theta| < {args.stop_deg:.0f} deg. Ctrl+C to abort.\n")

        try:
            while True:
                t0 = time.perf_counter()

                obs = reader.get()
                if obs is None:
                    ser.write(b"u 0.0\n")
                    time.sleep(0.005)
                    continue

                cos_th, sin_th, th_dot, phi, phi_dot = obs
                theta_deg = float(np.degrees(np.arctan2(sin_th, cos_th)))

                if abs(np.arctan2(sin_th, cos_th)) < stop_rad:
                    ser.write(b"u 0.0\n")
                    time.sleep(0.05)
                    ser.write(b"s\n")
                    print(f"\nReached upright! theta={theta_deg:+.1f}deg  th_dot={th_dot:+.1f} rad/s")
                    print("Waiting for pendulum to hang before next attempt...")
                    # Wait until hanging (|theta| > 160 deg) then prompt
                    while True:
                        obs2 = reader.get()
                        if obs2 is not None:
                            th2 = float(np.degrees(np.arctan2(obs2[1], obs2[0])))
                            print(f"  theta={th2:+6.1f}deg", end="\r")
                            if abs(th2) > 160.0:
                                print(f"\nHanging (theta={th2:+.1f}deg). Press Enter to swing again...")
                                input()
                                ser.reset_input_buffer()
                                break
                        time.sleep(0.02)

                u = swingup(obs)
                ser.write(f"u {u:.4f}\n".encode())
                energy = swingup.pendulum_energy(float(cos_th), float(th_dot))
                dE = swingup.E_ref - energy
                if dE < args.coast * swingup.E_max:
                    mode = "coast"
                else:
                    mode = "pump"

                print(f"theta={theta_deg:+6.1f}deg  phi={np.degrees(phi):+6.1f}deg  "
                      f"th_dot={th_dot:+5.1f}  dE={dE:+.4f}  {mode:5s}  u={u:+5.2f}", end="\r")

                wait = CONTROL_DT - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            ser.write(b"u 0.0\n")
            time.sleep(0.05)
            ser.write(b"s\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
