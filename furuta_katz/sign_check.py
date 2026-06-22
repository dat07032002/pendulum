"""
sign_check.py — verify the arm->pendulum coupling sign matches the sim model.
** MOVES THE ARM GENTLY (pendulum hanging). ** Flash nidec_velocity.ino first.

If balance "falls right away", the control sign is almost certainly flipped: the
arm pushes the pendulum the wrong way. This pulses a small +arm velocity at the
hanging rest and measures the sign of the pendulum's response.

Sim prediction (theta_ddot = alpha*sin - beta*cos*a, beta>0, at hanging cos=-1):
   a > 0  ->  theta_dot becomes POSITIVE.
So: a +v command should give +theta_dot. If hardware gives -theta_dot, the coupling
is flipped -> set plant.BETA negative (the controllers pick it up automatically).

Usage: python sign_check.py [--port COM5]
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import serial

import config

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
PULSE_V = 4.0
PULSE_S = 0.35


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


def main():
    ap = argparse.ArgumentParser(description="Check the coupling sign (MOVES THE ARM gently).")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    print("** sign_check pulses the arm. Let the pendulum HANG still first. **")
    with serial.Serial(args.port, config.BAUD, timeout=0.05) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()

        def send(v):
            ser.write(f"v {v:.3f}\n".encode()); ser.flush()

        def read():
            raw = ser.readline()
            if raw:
                return parse_obs(raw.decode("utf-8", errors="replace"))
            return None

        # confirm hanging
        send(0.0); time.sleep(0.3)
        base = None
        for _ in range(50):
            o = read()
            if o is not None:
                base = o
        if base is None or base[0] > -0.9:
            print(f"Pendulum not hanging (cos_theta={base[0] if base else None}). Let it settle and retry.")
            send(0.0); return

        # +v pulse, log response
        tds, pds = [], []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < PULSE_S:
            send(+PULSE_V)
            o = read()
            if o is not None:
                tds.append(o[2]); pds.append(o[4])
            time.sleep(config.CONTROL_DT)
        send(0.0); time.sleep(0.05); ser.write(b"s\n"); ser.flush()

    if not tds:
        print("No data captured."); return
    td = float(np.mean(tds)); pd = float(np.mean(pds))
    print(f"\n+v command -> mean phi_dot = {pd:+.2f} rad/s  (motor direction; expect +)")
    print(f"+v command -> mean theta_dot = {td:+.2f} rad/s  (coupling; sim expects +)")
    if abs(td) < 0.2:
        print("=> pendulum barely responded; increase PULSE_V and retry.")
    elif td > 0:
        print("=> coupling sign MATCHES the sim model. Keep plant.BETA = +0.70.")
        print("   (so 'falls right away' is not a coupling flip -- look at gains/deadzone.)")
    else:
        print("=> coupling sign is FLIPPED vs sim. Fix: set plant.BETA negative (-0.70),")
        print("   which flips the LQR and the swing-up direction consistently. Then retry.")
    if pd < 0:
        print("   NOTE: +v gave -phi_dot -> motor/encoder direction is also reversed.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
