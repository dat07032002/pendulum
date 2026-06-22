"""
balance_chip.py — monitor + tune the ON-CHIP balance (control runs in the firmware).

The ESP32 now runs the LQR+observer balance itself (no PC in the control loop, so no
serial latency). This script only WATCHES theta/phi and forwards your typed commands:
    bal                      arm balance (then lift the rod upright; it engages itself)
    k <kphi> <kth> <kphid> <kthd>   set LQR gains live (default -18.3 855.9 -9.1 61.2)
    bvmax <rad/s>            balance arm-speed cap
    s                        stop / disarm
    q                        quit this monitor

Flash nidec_velocity.ino first. Usage: python balance_chip.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def main():
    ap = argparse.ArgumentParser(description="Monitor/tune the on-chip balance.")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    ser = serial.Serial(args.port, config.BAUD, timeout=0.05)
    time.sleep(2.0)
    ser.reset_input_buffer()

    def forward():
        for line in sys.stdin:
            ser.write((line.strip() + "\n").encode("ascii"))
            ser.flush()
    threading.Thread(target=forward, daemon=True).start()

    print("ON-CHIP BALANCE monitor. Commands: bal | k a b c d | bvmax v | s | q")
    print("-> type 'bal', then LIFT the rod to upright; the firmware catches & holds it.")
    last = 0.0
    best_hold = 0.0
    engaged_at = None
    while True:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        m = OBS_RE.search(line)
        if m:
            p = m.group(1).split(",")
            if len(p) == 5:
                try:
                    theta = np.degrees(np.arctan2(float(p[1]), float(p[0])))
                    phi = np.degrees(float(p[3]))
                except ValueError:
                    continue
                now = time.time()
                # track how long it stays within 20 deg of upright (a "hold")
                if abs(theta) < 20.0:
                    engaged_at = engaged_at or now
                    best_hold = max(best_hold, now - engaged_at)
                else:
                    engaged_at = None
                if now - last > 0.25:
                    bar = "UP" if abs(theta) < 20 else ("near" if abs(theta) < 60 else "")
                    print(f"theta={theta:+6.1f}  phi={phi:+6.0f}  hold={best_hold:4.1f}s  {bar}   ",
                          end="\r", flush=True)
                    last = now
        elif line.startswith("#"):
            print("\n" + line)          # firmware messages: BALANCE engaged / lost / K set


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
