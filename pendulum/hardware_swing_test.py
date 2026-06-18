"""
Hardcoded back-and-forth swing test — no policy involved.

Alternates motor direction to build pendulum energy.
Safety stops on phi limit, phi_dot limit, or keyboard interrupt.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    print("Missing dependency: pyserial")
    print("Install: python -m pip install pyserial")
    sys.exit(1)

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def parse_obs(line: str) -> np.ndarray | None:
    match = OBS_RE.search(line)
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) != 5:
        return None
    try:
        return np.array([float(p) for p in parts], dtype=np.float32)
    except ValueError:
        return None


def send_u(port: serial.Serial, u: float) -> None:
    port.write(f"u {u:.5f}\n".encode("ascii"))
    port.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardcoded swing back-and-forth safety test.")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--power", type=float, default=0.2, help="Motor power 0-1")
    parser.add_argument("--half-period", type=float, default=0.4, help="Seconds per direction (ignored if --sensor-mode)")
    parser.add_argument("--cycles", type=int, default=10, help="Number of back-and-forth cycles")
    parser.add_argument("--phi-limit-deg", type=float, default=90.0, help="Stop if abs(phi) exceeds this")
    parser.add_argument("--phi-dot-limit", type=float, default=15.0, help="Stop if abs(phi_dot) exceeds this rad/s")
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to send u=0 before starting")
    parser.add_argument("--sensor-mode", action="store_true", help="Reverse arm based on pendulum swing direction (theta_dot sign) instead of fixed timer")
    args = parser.parse_args()

    phi_limit = np.radians(args.phi_limit_deg)
    power = abs(args.power)

    mode_str = "sensor-triggered (follows pendulum swing)" if args.sensor_mode else f"fixed timer (half-period={args.half_period:.2f}s)"
    print(f"Swing test: power=+/-{power:.2f}, mode={mode_str}, "
          f"cycles={args.cycles}, phi limit=+/-{args.phi_limit_deg:.0f}deg, "
          f"phi_dot limit=+/-{args.phi_dot_limit:.1f}rad/s")
    print("No policy — hardcoded alternating motor commands.")

    try:
        with serial.Serial(args.port, args.baud, timeout=0.01) as port:
            time.sleep(2.0)
            port.reset_input_buffer()
            send_u(port, 0.0)

            print(f"Warmup {args.warmup:.1f}s ...")
            t_start = time.perf_counter()
            while time.perf_counter() - t_start < args.warmup:
                port.readline()

            print("Starting swings. Ctrl+C to stop.")

            direction = 1.0
            cycle = 0
            t_phase = time.perf_counter()
            last_print = 0.0
            stopped = False
            last_theta_dot = 0.0

            while cycle < args.cycles:
                now = time.perf_counter()

                # Switch direction each half-period (timer mode)
                if not args.sensor_mode and now - t_phase >= args.half_period:
                    direction *= -1.0
                    t_phase = now
                    cycle += (1 if direction == 1.0 else 0)

                raw = port.readline()
                if not raw:
                    send_u(port, direction * power)
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                obs = parse_obs(line)
                if obs is None:
                    continue

                phi = float(obs[3])
                phi_dot = float(obs[4])
                theta = float(np.arctan2(obs[1], obs[0]))
                theta_dot = float(obs[2])

                # Sensor mode: arm direction follows pendulum swing direction
                if args.sensor_mode:
                    new_dir = 1.0 if theta_dot >= 0 else -1.0
                    if new_dir != direction:
                        direction = new_dir
                        cycle += 1
                    last_theta_dot = theta_dot

                if abs(phi) > phi_limit:
                    print(f"Safety stop: phi={np.degrees(phi):.1f}deg exceeds limit.")
                    stopped = True
                    break
                if abs(phi_dot) > args.phi_dot_limit:
                    print(f"Safety stop: phi_dot={phi_dot:.2f} rad/s exceeds limit.")
                    stopped = True
                    break

                send_u(port, direction * power)

                if now - last_print >= 0.15:
                    print(
                        f"cycle={cycle:2d} dir={'+' if direction > 0 else '-'} "
                        f"theta={np.degrees(theta):+7.2f}deg "
                        f"phi={np.degrees(phi):+7.2f}deg "
                        f"phi_dot={phi_dot:+6.2f} "
                        f"u={direction * power:+.2f}"
                    )
                    last_print = now

            send_u(port, 0.0)
            time.sleep(0.05)
            send_u(port, 0.0)

            if not stopped:
                print(f"Done — completed {args.cycles} cycles.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        try:
            if "port" in locals() and port.is_open:
                send_u(port, 0.0)
        except Exception:
            pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
