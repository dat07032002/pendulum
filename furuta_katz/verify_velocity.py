"""
verify_velocity.py — check the velocity firmware tracks commanded arm speeds.
** MOVES THE ARM. ** Flash firmware/nidec_velocity/nidec_velocity.ino first, hand-center.

For each target speed it pre-positions to one side, then commands "v target" while
sweeping across center, and reports the achieved |phi_dot| vs the target. Good
tracking (achieved ~ target) means the duty map + break-free kick work, and hw_env
can send velocity setpoints directly.

Usage: python verify_velocity.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
START_DEG, END_DEG, SOFT_DEG = 55.0, 35.0, 80.0
TARGETS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]


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
        self.ser.write(f"v {v:.4f}\n".encode("ascii"))
        self.ser.flush()

    def send_zero(self):
        self.ser.write(b"z\n"); self.ser.flush()

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
            self.send_v(0.0); time.sleep(0.05)
            self.ser.write(b"s\n"); self.ser.flush()
        finally:
            self.ser.close()


def drive_to(link, target_rad, soft_rad, v_pos=3.0, timeout=8.0):
    hard = np.deg2rad(config.PHI_LIMIT_DEG)
    t0 = time.perf_counter()
    phi = 0.0
    while time.perf_counter() - t0 < timeout:
        o = link.read_obs()
        if o is None:
            continue
        phi = o[3]
        direction = np.sign(target_rad - phi)
        if abs(phi) > hard:
            link.send_v(0.0); return False
        if abs(phi - target_rad) < np.deg2rad(10.0):
            link.send_v(0.0); return True
        if abs(phi) > soft_rad and np.sign(phi) == direction:
            link.send_v(0.0); return False
        link.send_v(v_pos * direction)
        time.sleep(config.CONTROL_DT)
    link.send_v(0.0)
    return abs(phi - target_rad) < np.deg2rad(18.0)


def sweep(link, v_signed, end_rad, soft_rad):
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
        if abs(phi) > soft_rad:
            break
        if np.sign(v_signed) * phi > end_rad:
            break
        if time.perf_counter() - t0 > 0.6 and abs(phi - phi0) < np.deg2rad(4.0):
            link.send_v(0.0); return None
        link.send_v(v_signed)
        speeds.append(abs(phi_dot))
        time.sleep(config.CONTROL_DT)
    link.send_v(0.0)
    if len(speeds) < 4:
        return None
    return float(np.median(speeds[len(speeds) // 2:]))


def main():
    ap = argparse.ArgumentParser(description="Verify velocity tracking (MOVES THE ARM).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()
    soft = np.deg2rad(SOFT_DEG)
    end = np.deg2rad(END_DEG)
    start = np.deg2rad(START_DEG)

    print("** verify_velocity MOVES THE ARM. Hand-center first (cable neutral). **")
    link = Link(args.port, config.BAUD)
    results = []
    try:
        link.send_zero(); time.sleep(0.2)
        for i, vt in enumerate(TARGETS):
            sign = 1.0 if i % 2 == 0 else -1.0
            if not drive_to(link, -sign * start, soft):
                print(f"   v={vt:4.1f}  ->  could not pre-position; skipping")
                drive_to(link, 0.0, soft)
                continue
            achieved = sweep(link, sign * vt, end, soft)
            if achieved is None:
                print(f"   v={vt:4.1f}  ->  no motion")
            else:
                err = 100.0 * (achieved - vt) / vt
                print(f"   commanded v={vt:5.1f}  ->  achieved {achieved:5.2f} rad/s  ({err:+.0f}%)")
                results.append((vt, achieved))
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        try:
            drive_to(link, 0.0, soft, timeout=5.0)
        except Exception:
            pass
        link.stop()

    if results:
        print("\n===== velocity tracking =====")
        for vt, a in results:
            print(f"  target {vt:5.1f}  ->  {a:5.2f} rad/s")
        print("Good tracking (within ~25%) means the firmware map works and hw_env")
        print("can send velocity setpoints directly.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
