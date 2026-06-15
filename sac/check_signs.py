"""
One-shot sign/unit consistency check for the Furuta hardware obs.

Pulses the motor (+u then -u), records the obs trace, and verifies:
  - phi_dot agrees with d(phi)/dt        (sign + rad/s scale)
  - theta_dot agrees with d(theta)/dt    (sign + rad/s scale)
  - which way the arm moves for +u       (informational; SAC learns either)

Sends "u 0" on every exit path.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import serial

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def parse_obs(line: str):
    match = OBS_RE.search(line)
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) != 5:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def send_u(port: serial.Serial, u: float) -> None:
    port.write(f"u {u:.5f}\n".encode("ascii"))
    port.flush()


def record(port: serial.Serial, seconds: float, samples: list) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        raw = port.readline()
        if not raw:
            continue
        obs = parse_obs(raw.decode("utf-8", errors="replace").strip())
        if obs is not None:
            samples.append((time.perf_counter(), *obs))


def analyze(samples: list, label: str) -> bool:
    if len(samples) < 30:
        print(f"[{label}] Not enough samples ({len(samples)}); cannot analyze.")
        return False

    arr = np.array(samples)
    t = arr[:, 0]
    cos_t, sin_t, theta_dot, phi, phi_dot = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]

    theta = np.unwrap(np.arctan2(sin_t, cos_t))
    d_theta = np.gradient(theta, t)
    d_phi = np.gradient(phi, t)

    ok = True
    for name, reported, derived in (("phi", phi_dot, d_phi), ("theta", theta_dot, d_theta)):
        mask = np.abs(derived) > 0.2  # only judge while actually moving
        if mask.sum() < 10:
            print(f"[{label}] {name}: too little motion to judge "
                  f"(max |d{name}/dt| = {np.abs(derived).max():.2f} rad/s)")
            ok = False
            continue
        corr = float(np.corrcoef(reported[mask], derived[mask])[0, 1])
        scale = float(np.dot(reported[mask], derived[mask]) / np.dot(derived[mask], derived[mask]))
        verdict = "OK" if corr > 0.8 and 0.5 < scale < 2.0 else "PROBLEM"
        if verdict == "PROBLEM":
            ok = False
        print(f"[{label}] {name}_dot vs d({name})/dt: corr={corr:+.3f} scale={scale:+.3f} -> {verdict}")
        if corr < -0.5:
            print(f"          -> {name}_dot has the WRONG SIGN relative to {name}.")
        elif corr > 0.8 and not (0.5 < scale < 2.0):
            print(f"          -> sign is right but units look off (scale {scale:.2f}; expected ~1 for rad/s).")

    print(f"[{label}] arm moved {np.degrees(phi[-1] - phi[0]):+.1f} deg "
          f"(phi {phi[0]:+.3f} -> {phi[-1]:+.3f} rad), "
          f"peak |theta swing| {np.degrees(np.abs(theta - theta[0]).max()):.1f} deg")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Pulse the motor and verify obs sign/unit consistency.")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--u", type=float, default=0.35, help="Pulse magnitude (must exceed motor deadband ~0.235)")
    parser.add_argument("--duration", type=float, default=0.35, help="Pulse length in seconds")
    parser.add_argument("--settle", type=float, default=4.0, help="Seconds to wait between pulses")
    args = parser.parse_args()

    print(f"Opening {args.port} at {args.baud} baud")
    print(f"Will pulse u={args.u:+.2f} then u={-args.u:+.2f} for {args.duration:.2f}s each.")

    results = []
    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as port:
            time.sleep(2.0)
            port.reset_input_buffer()
            send_u(port, 0.0)

            baseline: list = []
            record(port, 1.0, baseline)
            if baseline:
                b = np.array(baseline)[:, 1:].mean(axis=0)
                print(f"\nBaseline (motor off): cos={b[0]:+.3f} sin={b[1]:+.3f} "
                      f"theta_dot={b[2]:+.3f} phi={b[3]:+.3f} phi_dot={b[4]:+.3f}")
                if b[0] > -0.9:
                    print("WARNING: cos(theta) is not ~ -1 at rest. "
                          "Pendulum is not hanging or the AS5600 zero is off!")
            else:
                print("ERROR: no observations received. Check the serial link.")
                return 1

            for sign in (+1.0, -1.0):
                u = sign * abs(args.u)
                label = f"pulse {u:+.2f}"
                print(f"\n--- {label} ---")
                samples: list = []
                send_u(port, u)
                record(port, args.duration, samples)
                send_u(port, 0.0)
                record(port, 1.5, samples)  # capture the reaction swing too
                send_u(port, 0.0)
                results.append(analyze(samples, label))
                time.sleep(args.settle)
                port.reset_input_buffer()

            send_u(port, 0.0)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        try:
            if "port" in locals() and port.is_open:
                send_u(port, 0.0)
        except Exception:
            pass
        return 1
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        return 1

    print("\n=== RESULT:", "ALL CHECKS PASSED" if all(results) and results else "PROBLEMS FOUND", "===")
    return 0 if all(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
