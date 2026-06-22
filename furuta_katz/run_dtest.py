"""
run_dtest.py — trigger the firmware's steady raw-duty test. ** MOVES THE ARM. **

Sends "dtest"; the firmware holds each raw duty across a long sweep and reports the
SETTLED center speed. This tells us whether low duty gives low speed (the motor can
do the fine control balance needs) or coasts fast regardless (it cannot).

Flash the updated nidec_velocity.ino, hand-center the arm, then:
    python run_dtest.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import time

import serial

import config

D_RE = re.compile(r"duty=([0-9.]+)\s+(?:steady|settled)=([0-9.]+)")


def main():
    ap = argparse.ArgumentParser(description="Run the firmware steady-duty test (MOVES THE ARM).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("** firmware will hold each duty and measure settled speed. Hand-center first. **")
    pairs = []
    with serial.Serial(args.port, config.BAUD, timeout=0.2) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.write(b"dtest\n"); ser.flush()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 120.0:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            print("   " + line)
            m = D_RE.search(line)
            if m:
                pairs.append((float(m.group(1)), float(m.group(2))))
            if line == "DONE" or "ABORT" in line:
                break
        ser.write(b"s\n"); ser.flush()

    if pairs:
        print("\n===== raw duty -> settled center speed =====")
        for d, s in pairs:
            print(f"  duty={d:.3f}  ->  {s:5.2f} rad/s")
        lo = pairs[0][1]
        print(f"\nlowest duty (0.040) settled speed: {lo:.2f} rad/s")
        if lo < 3.0:
            print("-> motor CAN go slow; vtest was just kick momentum. Fix: kick only from rest.")
        else:
            print("-> even low duty stays fast: the driver coasts / won't hold low speed.")
            print("   We'd adapt the strategy (the integrated driver lacks fine low-speed control).")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
