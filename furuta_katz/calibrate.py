"""
calibrate.py — measure the motor's velocity-command scale.  ** MOVES THE MOTOR **

Measures the two numbers hw_env needs to turn the controller's arm-speed setpoint
into the firmware command u in [-1, 1]:

  U_MIN : smallest |u| that actually moves the arm (the speed-loop stall floor).
  V_MAX : arm speed [rad/s] the motor produces at |u| = 1.

The arm is fast and the cable travel is limited, so we don't try to hold a steady
spin. Instead, for each test level we PRE-POSITION the arm to one side, then SWEEP
across the center while logging |phi_dot|, stopping before the soft limit. The
steady part of the sweep gives the speed. Direction alternates each level so the
cable does not wind net. Fit speed = m*|u| + b over the moving levels and
extrapolate V_MAX = m + b (driving to u=1 directly would slam the cable limit).

Safety:
  - never commands more than --max-u (default 0.45)
  - unwinds to center at startup and aborts any motion past the soft limit
  - sends "u 0" on every exit, including Ctrl-C / errors
  - requires you to type "go" before anything moves (skip with --yes)

Run attended, with the rig clear and the cable free:
    python calibrate.py [--port COM5] [--max-u 0.45]
Results are written to calibration.json (read automatically by config.py).
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")

POS_U = 0.25        # drive level for positioning/recentering (above the ~0.15 stall floor)
START_DEG = 72.0    # pre-position the arm this far to one side before a sweep
END_DEG = 40.0      # stop the sweep here -- the arm is FAST, leave big overshoot margin


def parse_obs(line: str):
    m = OBS_RE.search(line)
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) != 5:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


class Link:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baud, timeout=0.02)
        time.sleep(2.0)             # ESP32 resets on open
        self.ser.reset_input_buffer()
        self.send_u(0.0)

    def send_u(self, u: float):
        self.ser.write(f"u {u:.5f}\n".encode("ascii"))
        self.ser.flush()

    def send_zero(self):
        self.ser.write(b"z\n")
        self.ser.flush()

    def read_obs(self, timeout=0.2):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            o = parse_obs(raw.decode("utf-8", errors="replace"))
            if o is not None:
                return o
        return None

    def stop(self):
        try:
            self.send_u(0.0)
            time.sleep(0.05)
            self.send_u(0.0)
        finally:
            self.ser.close()


def drive_to(link: Link, target_rad: float, soft_rad: float, timeout: float = 8.0) -> bool:
    """Drive the arm toward target_rad at POS_U; stop on arrival.

    Always allows INWARD motion (so it can recover an arm left past the soft
    limit); only refuses to push further OUTWARD past soft. Hard-stops at the
    cable limit."""
    hard_rad = np.deg2rad(config.PHI_LIMIT_DEG)
    t0 = time.perf_counter()
    phi = 0.0
    while time.perf_counter() - t0 < timeout:
        o = link.read_obs()
        if o is None:
            continue
        phi = o[3]
        direction = np.sign(target_rad - phi)
        if abs(phi) > hard_rad:                        # hard cable limit
            link.send_u(0.0)
            return False
        if abs(phi - target_rad) < np.deg2rad(12.0):   # arrived
            link.send_u(0.0)
            return True
        if abs(phi) > soft_rad and np.sign(phi) == direction:  # would push further out
            link.send_u(0.0)
            return False
        link.send_u(POS_U * direction)
        time.sleep(config.CONTROL_DT)
    link.send_u(0.0)
    return abs(phi - target_rad) < np.deg2rad(20.0)


def sweep_speed(link: Link, u_signed: float, end_rad: float, soft_rad: float):
    """Sweep at u_signed across center to end_rad; return steady median |phi_dot|.

    Returns None if the arm never moved (below the stall floor)."""
    speeds = []
    phi0 = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.5:
        o = link.read_obs()
        if o is None:
            continue
        phi, phi_dot = o[3], o[4]
        if phi0 is None:
            phi0 = phi
        if abs(phi) > soft_rad:                        # safety stop
            break
        if np.sign(u_signed) * phi > end_rad:          # reached the far side
            break
        # early-out: not moving after 0.5 s -> below stall floor
        if time.perf_counter() - t0 > 0.5 and abs(phi - phi0) < np.deg2rad(5.0):
            link.send_u(0.0)
            return None
        link.send_u(u_signed)
        speeds.append(abs(phi_dot))
        time.sleep(config.CONTROL_DT)
    link.send_u(0.0)
    if len(speeds) < 4:
        return None
    steady = speeds[len(speeds) // 2:]                 # back half = up to speed
    return float(np.median(steady))


def main():
    ap = argparse.ArgumentParser(description="Measure U_MIN and V_MAX (MOVES THE MOTOR).")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--max-u", type=float, default=0.20, help="largest |u| to test (the arm is fast; keep low)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    soft_rad = np.deg2rad(config.PHI_SOFT_DEG)
    end_rad = np.deg2rad(END_DEG)
    start_rad = np.deg2rad(START_DEG)
    levels = [round(u, 3) for u in np.arange(0.06, args.max_u + 1e-6, 0.02)]

    print("** calibrate.py MOVES THE MOTOR. Keep clear. **")
    print("   FIRST, by hand: center the arm so the cable is NEUTRAL (un-wound).")
    print("   The arm encoder is incremental, so the script treats this start")
    print(f"   position as center and sweeps within +-{config.PHI_SOFT_DEG:.0f} deg of it.")
    print(f"   port={args.port}  test |u|={levels}")
    if not args.yes:
        if input('   cable neutral? type "go" to start: ').strip().lower() != "go":
            print("aborted.")
            return

    link = Link(args.port, config.BAUD)
    measured = []  # (u, speed)
    try:
        link.send_zero()        # define the (hand-centered) boot position as phi = 0
        time.sleep(0.2)

        for i, u in enumerate(levels):
            sign = 1.0 if (i % 2 == 0) else -1.0       # alternate sweep direction
            if not drive_to(link, -sign * start_rad, soft_rad):
                print(f"   u={u:4.2f}  ->  could not pre-position; skipping")
                drive_to(link, 0.0, soft_rad, timeout=4.0)
                continue
            spd = sweep_speed(link, sign * u, end_rad, soft_rad)
            if spd is None:
                print(f"   u={u:4.2f}  ->  did not move (below stall floor)")
            else:
                print(f"   u={u:4.2f}  ->  |phi_dot| = {spd:5.2f} rad/s")
                measured.append((u, spd))
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        # Always try to bring the arm back to center before releasing the motor,
        # so a failed/aborted run never leaves the cable wound.
        try:
            print("   recentering before exit...")
            drive_to(link, 0.0, soft_rad, timeout=5.0)
        except Exception:
            pass
        link.stop()

    moving = [(u, s) for (u, s) in measured if s > 0.5]
    if len(moving) < 2:
        print("\nNot enough moving samples to fit. Raise --max-u and retry, attended.")
        return
    u_arr = np.array([u for u, _ in moving])
    s_arr = np.array([s for _, s in moving])
    m, b = np.polyfit(u_arr, s_arr, 1)
    v_max = float(m * 1.0 + b)
    u_min = float(min(u for u, _ in moving))

    print("\n===== calibration result =====")
    print(f"linear fit: |phi_dot| ~= {m:.2f}*|u| + {b:.2f}")
    print(f"U_MIN (first moving level) : {u_min:.3f}")
    print(f"V_MAX (speed at |u|=1, extrapolated): {v_max:.2f} rad/s")
    path = config.save_calibration(v_max, u_min,
                                   extra={"fit_slope": float(m), "fit_intercept": float(b),
                                          "samples": measured})
    print(f"saved -> {path}  (config.py loads it automatically)")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
        sys.exit(1)
