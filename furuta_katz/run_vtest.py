"""
run_vtest.py — trigger the velocity firmware's built-in self-test. ** MOVES THE ARM. **

Sends "vtest" to nidec_velocity.ino; the firmware ping-pongs the arm and measures
the steady arm speed (at center crossing) for each commanded velocity, with no
serial latency. Prints the commanded-vs-measured table.

Flash nidec_velocity.ino, hand-center the arm, then run:
    python run_vtest.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import time

import serial

import config

V_RE = re.compile(r"v=([0-9.]+)\s+measured=([0-9.]+)")


def main():
    ap = argparse.ArgumentParser(description="Run the firmware velocity self-test (MOVES THE ARM).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("** firmware will ping-pong the arm and measure speed. Hand-center first. **")
    pairs = []
    with serial.Serial(args.port, config.BAUD, timeout=0.2) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.write(b"vtest\n"); ser.flush()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 90.0:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            print("   " + line)
            m = V_RE.search(line)
            if m:
                pairs.append((float(m.group(1)), float(m.group(2))))
            if line == "DONE" or "ABORT" in line:
                break
        ser.write(b"s\n"); ser.flush()

    if pairs:
        print("\n===== commanded vs measured =====")
        for vc, vm in pairs:
            err = 100.0 * (vm - vc) / vc
            print(f"  command {vc:5.1f}  ->  measured {vm:5.2f} rad/s  ({err:+.0f}%)")
        mono = all(pairs[i][1] <= pairs[i + 1][1] + 0.5 for i in range(len(pairs) - 1))
        print(f"\nmonotonic: {'yes' if mono else 'NO'}  "
              f"(yes -> speed rises with command; we can rescale the curve for exact tracking)")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
