"""
run_pwm_sweep.py — trigger diag_pwm_sweep.ino and collect the duty->speed curve.

Sends "go" to the diagnostic firmware, prints every line it returns, and saves the
parsed (duty, speed) points to pwm_curve.json. ** The firmware MOVES THE ARM. **
Flash firmware/diag_pwm_sweep/diag_pwm_sweep.ino first and hand-center the arm.

Usage:
    python run_pwm_sweep.py [--port COM5]
"""
from __future__ import annotations

import argparse
import json
import re
import time

import serial

import config

LINE_RE = re.compile(r"duty=([0-9.]+)\s+speed=([0-9.]+)")
STALL_RE = re.compile(r"duty=([0-9.]+)\s+STALLED")


def main():
    ap = argparse.ArgumentParser(description="Run the raw-duty sweep (MOVES THE ARM).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("** firmware will sweep raw PWM duty and spin the arm. Keep clear. **")
    print("   (flash diag_pwm_sweep.ino, hand-center the arm so the cable is neutral)")

    points = []      # (duty, speed) ; speed None = stalled
    with serial.Serial(args.port, config.BAUD, timeout=0.2) as ser:
        time.sleep(2.0)               # ESP32 resets on open
        ser.reset_input_buffer()
        ser.write(b"go\n")
        ser.flush()

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 120.0:     # generous; sweep self-terminates
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            print("   " + line)
            m = LINE_RE.search(line)
            if m:
                points.append((float(m.group(1)), float(m.group(2))))
                continue
            s = STALL_RE.search(line)
            if s:
                points.append((float(s.group(1)), None))
                continue
            if line == "DONE" or "ABORTED" in line:
                break
        # safety: tell the motor to stop in case the sweep was interrupted
        ser.write(b"s\n")
        ser.flush()

    moving = [(d, s) for d, s in points if s is not None]
    print("\n===== raw duty -> arm speed =====")
    for d, s in points:
        print(f"  duty={d:.3f}  ->  {'STALLED' if s is None else f'{s:.2f} rad/s'}")
    if moving:
        d0, s0 = moving[0]
        print(f"\nslowest moving level : duty={d0:.3f} -> {s0:.2f} rad/s")
        print(f"fastest tested level : duty={moving[-1][0]:.3f} -> {moving[-1][1]:.2f} rad/s")
        print("-> if low duties give slow speeds, we can build a fine speed map.")
        print("   if it stalls then jumps straight to ~13 rad/s, the motor itself")
        print("   has no slow regime and we adapt the control strategy instead.")

    with open("pwm_curve.json", "w") as f:
        json.dump({"points": points}, f, indent=2)
    print("\nsaved -> pwm_curve.json")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
