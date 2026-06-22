"""
track_test.py — clean velocity tracking: long sweeps, measure speed AT CENTER.
** MOVES THE ARM. ** Flash nidec_velocity.ino, hand-center first.

The short-oscillation test was dominated by the break-free kick and momentum. Here
each sweep runs the full +-70 deg so the arm settles, and we sample |phi_dot| only
in the center window (|phi| < 30 deg) where it is at steady speed and the cable
spring load is small. That isolates the firmware's true duty->speed tracking.

Usage: python track_test.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
TARGETS = [2.0, 4.0, 6.0, 8.0, 10.0]
FAR_DEG, SOFT_DEG, WIN_DEG = 70.0, 82.0, 30.0


def parse_obs(line):
    m = OBS_RE.search(line)
    if not m:
        return None
    p = m.group(1).split(",")
    if len(p) != 5:
        return None
    try:
        return [float(x) for x in p]
    except ValueError:
        return None


class Link:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.02)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.send_v(0.0)

    def send_v(self, v):
        self.ser.write(f"v {v:.4f}\n".encode("ascii")); self.ser.flush()

    def send_zero(self):
        self.ser.write(b"z\n"); self.ser.flush()

    def read_obs(self, timeout=0.05):
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
            self.send_v(0.0); time.sleep(0.05)
            self.ser.write(b"s\n"); self.ser.flush()
        finally:
            self.ser.close()


def sweep_center(link, v_signed, far_rad, soft_rad, win_rad):
    """Sweep to the far side; collect |phi_dot| while crossing center. Median or None."""
    speeds = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        o = link.read_obs()
        if o is None:
            continue
        phi, phi_dot = o[3], o[4]
        if abs(phi) > soft_rad:
            break
        if np.sign(v_signed) * phi > far_rad:
            break
        if abs(phi) < win_rad and time.perf_counter() - t0 > 0.1:
            speeds.append(abs(phi_dot))
        link.send_v(v_signed)
        time.sleep(config.CONTROL_DT)
    link.send_v(0.0)
    return float(np.median(speeds)) if len(speeds) >= 5 else None


def main():
    ap = argparse.ArgumentParser(description="Clean velocity tracking via center-crossing.")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()
    far, soft, win = np.deg2rad(FAR_DEG), np.deg2rad(SOFT_DEG), np.deg2rad(WIN_DEG)

    print("** track_test MOVES THE ARM (sweeps +-70 deg). Hand-center first. **")
    link = Link(args.port, config.BAUD)
    results = []
    try:
        link.send_zero(); time.sleep(0.2)
        # start by going to one side
        sweep_center(link, -4.0, far, soft, win)
        for vt in TARGETS:
            samples = []
            for _ in range(3):
                # arm is on the - side: sweep + (measure), then - (measure), back to - side
                s1 = sweep_center(link, +vt, far, soft, win)
                s2 = sweep_center(link, -vt, far, soft, win)
                if s1 is not None:
                    samples.append(s1)
                if s2 is not None:
                    samples.append(s2)
            if samples:
                med = float(np.median(samples))
                err = 100.0 * (med - vt) / vt
                print(f"   target v={vt:5.1f}  ->  achieved {med:5.2f} rad/s  ({err:+.0f}%)  [n={len(samples)}]")
                results.append((vt, med))
            else:
                print(f"   target v={vt:5.1f}  ->  no clean center samples")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        try:
            sweep_center(link, 4.0, np.deg2rad(5.0), soft, win)  # nudge back toward center
        except Exception:
            pass
        link.stop()

    if results:
        print("\n===== steady velocity tracking (center crossing) =====")
        for vt, a in results:
            print(f"  target {vt:5.1f}  ->  {a:5.2f} rad/s")
        mono = all(results[i][1] <= results[i + 1][1] + 0.5 for i in range(len(results) - 1))
        print(f"monotonic with command: {'yes' if mono else 'NO'}  "
              f"(yes -> map is usable; we can refine the scale for exact tracking)")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
