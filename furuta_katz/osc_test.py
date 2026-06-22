"""
osc_test.py — fair velocity-tracking test: oscillate the arm gently AROUND CENTER.
** MOVES THE ARM. ** Flash nidec_velocity.ino, hand-center the arm first.

Avoids the cable-spring-loaded extremes. For each target speed it commands a +v/-v
square wave around center (half-period sized so travel stays ~+-40 deg), skips the
break-free transient, and reports the median achieved |phi_dot| per direction. This
shows whether the firmware tracks small velocity commands the way balance needs.

Usage: python osc_test.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
TARGETS = [2.0, 4.0, 6.0, 8.0]
TRAVEL_DEG = 40.0          # aim each half-swing to cover ~this far
SAFE_DEG = 75.0            # flip early if |phi| exceeds this


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


def half_swing(link, v_signed, t_half, safe_rad):
    """Drive v_signed for t_half; return median |phi_dot| after the kick transient."""
    speeds = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < t_half:
        o = link.read_obs()
        if o is None:
            continue
        phi, phi_dot = o[3], o[4]
        if abs(phi) > safe_rad:                 # too far: bail this half early
            break
        link.send_v(v_signed)
        if time.perf_counter() - t0 > 0.06:     # skip the break-free kick / accel
            speeds.append(abs(phi_dot))
    return float(np.median(speeds)) if len(speeds) >= 3 else None


def main():
    ap = argparse.ArgumentParser(description="Oscillate around center to test velocity tracking.")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()
    safe = np.deg2rad(SAFE_DEG)

    print("** osc_test MOVES THE ARM (gentle oscillation near center). Hand-center first. **")
    link = Link(args.port, config.BAUD)
    results = []
    try:
        link.send_zero(); time.sleep(0.2)
        for vt in TARGETS:
            t_half = float(np.clip(np.deg2rad(TRAVEL_DEG) / vt, 0.08, 0.5))
            got = []
            for cyc in range(4):                # a few cycles per target
                s_pos = half_swing(link, +vt, t_half, safe)
                s_neg = half_swing(link, -vt, t_half, safe)
                if s_pos is not None:
                    got.append(s_pos)
                if s_neg is not None:
                    got.append(s_neg)
            link.send_v(0.0); time.sleep(0.2)
            if got:
                med = float(np.median(got))
                err = 100.0 * (med - vt) / vt
                print(f"   target v={vt:4.1f}  ->  achieved {med:5.2f} rad/s  ({err:+.0f}%)  "
                      f"[n={len(got)}, half={t_half*1e3:.0f}ms]")
                results.append((vt, med))
            else:
                print(f"   target v={vt:4.1f}  ->  no clean samples")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        link.stop()

    if results:
        print("\n===== velocity tracking (near center) =====")
        for vt, a in results:
            print(f"  target {vt:4.1f}  ->  {a:5.2f} rad/s")
        good = sum(abs(a - vt) / vt < 0.30 for vt, a in results)
        print(f"{good}/{len(results)} within 30%. Good tracking -> firmware map works, "
              f"hw_env can send velocity setpoints.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
