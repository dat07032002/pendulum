"""
Interactive raw-PWM motor identification.

For each PWM test:
  1. Physically center the shoulder arm.
  2. Press Enter.
  3. Script sends s, z, then p <PWM>.
  4. Script captures stop reason, elapsed time, and final phi.
  5. Repeat after you manually recenter.

Requires the Arduino raw PWM firmware with commands:
  s, z, p <signed_pwm>
and output containing:
  phi_deg=...
  # Stop reason: ..., elapsed_ms=...
  # MOTOR STOP phi_deg=...
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import serial


PHI_RE = re.compile(r"phi_deg=([-+0-9.]+)")
STOP_RE = re.compile(r"# Stop reason: ([^,]+), elapsed_ms=([0-9]+)")
MOTOR_STOP_RE = re.compile(r"# MOTOR STOP phi_deg=([-+0-9.]+)")


def send(port: serial.Serial, cmd: str) -> None:
    print(f"SEND {cmd}")
    port.write((cmd + "\n").encode("ascii"))
    port.flush()


def run_one(port: serial.Serial, pwm: int, capture_seconds: float) -> dict:
    lines: list[str] = []
    phis: list[float] = []
    stop_reason = None
    elapsed_ms = None
    stop_phi = None

    send(port, "s")
    time.sleep(0.15)
    send(port, "z")
    time.sleep(0.2)
    port.reset_input_buffer()
    send(port, f"p {pwm}")

    start = time.perf_counter()
    while time.perf_counter() - start < capture_seconds:
        raw = port.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="replace").strip()
        lines.append(line)

        phi_match = PHI_RE.search(line)
        if phi_match:
            phis.append(float(phi_match.group(1)))

        stop_match = STOP_RE.search(line)
        if stop_match:
            stop_reason = stop_match.group(1)
            elapsed_ms = int(stop_match.group(2))

        motor_stop_match = MOTOR_STOP_RE.search(line)
        if motor_stop_match:
            stop_phi = float(motor_stop_match.group(1))

    send(port, "s")
    time.sleep(0.1)

    return {
        "pwm": pwm,
        "elapsed_ms": elapsed_ms,
        "stop_reason": stop_reason,
        "stop_phi_deg": stop_phi,
        "final_phi_deg": phis[-1] if phis else None,
        "max_abs_phi_deg": max((abs(v) for v in phis), default=None),
        "samples": len(phis),
        "tail": lines[-8:],
    }


def print_result(result: dict) -> None:
    print("Result:")
    print(f"  pwm              : {result['pwm']}")
    print(f"  stop_reason      : {result['stop_reason']}")
    print(f"  elapsed_ms       : {result['elapsed_ms']}")
    print(f"  stop_phi_deg     : {result['stop_phi_deg']}")
    print(f"  final_phi_deg    : {result['final_phi_deg']}")
    print(f"  max_abs_phi_deg  : {result['max_abs_phi_deg']}")
    print(f"  samples          : {result['samples']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive raw-PWM Furuta motor ID.")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--pwms", default="60,70,80,90", help="Comma-separated positive PWM values")
    parser.add_argument("--directions", default="+,-", help="Use '+', '-', or '+,-'")
    parser.add_argument("--capture-seconds", type=float, default=1.5)
    parser.add_argument("--out", default="motor_pwm_id.csv")
    args = parser.parse_args()

    pwm_values = [int(item.strip()) for item in args.pwms.split(",") if item.strip()]
    directions = [item.strip() for item in args.directions.split(",") if item.strip()]
    signed_pwms = []
    for pwm in pwm_values:
        for direction in directions:
            signed_pwms.append(pwm if direction == "+" else -pwm)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    results = []
    print(f"Opening {args.port}")
    print("For each test, physically center the arm, then press Enter.")
    print("The script sends s, z, then p <PWM>. It sends s after each capture.")

    with serial.Serial(args.port, args.baud, timeout=0.01) as port:
        time.sleep(2.0)
        port.reset_input_buffer()
        send(port, "s")

        for pwm in signed_pwms:
            input(f"\nCenter the arm physically for PWM {pwm}, then press Enter...")
            result = run_one(port, pwm, args.capture_seconds)
            print_result(result)
            results.append(result)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pwm",
                "elapsed_ms",
                "stop_reason",
                "stop_phi_deg",
                "final_phi_deg",
                "max_abs_phi_deg",
                "samples",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})

    print(f"\nSaved CSV -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
