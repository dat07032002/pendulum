"""passive_read.py — READ ONLY. Opens the serial link and prints theta/phi from the
obs stream for a few seconds. Sends NO commands, so the motor never moves. Used to
verify the link and that theta reads ~0 deg when the rod is held upright."""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0


def main():
    ser = serial.Serial(config.PORT, config.BAUD, timeout=0.1)
    time.sleep(2.0)
    ser.reset_input_buffer()
    t0 = time.time()
    thetas = []
    n = 0
    while time.time() - t0 < SECONDS:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        m = OBS_RE.search(line)
        if not m:
            if line.startswith("#"):
                print(line)
            continue
        p = m.group(1).split(",")
        if len(p) != 5:
            continue
        try:
            cos_t, sin_t, theta_dot, phi, phi_dot = (float(x) for x in p)
        except ValueError:
            continue
        theta = np.degrees(np.arctan2(sin_t, cos_t))
        thetas.append(theta)
        n += 1
        if n % 25 == 0:
            print(f"theta={theta:+6.1f} deg   phi={np.degrees(phi):+6.1f} deg   "
                  f"theta_dot={theta_dot:+5.2f}")
    ser.close()
    if thetas:
        a = np.array(thetas)
        print(f"\n--- {n} samples ---")
        print(f"theta: mean={a.mean():+.1f}  min={a.min():+.1f}  max={a.max():+.1f} deg")
        print("If you held it upright, mean theta should be near 0. A steady offset "
              "(e.g. +8 deg) = the upright zero is off -> fix with 'calup'.")
    else:
        print("No obs received. Is COM5 the ESP32 and is nidec_velocity.ino flashed?")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nClose any open Serial Monitor / balance_chip.py first.")
