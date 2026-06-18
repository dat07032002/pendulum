"""
Safe ESP32 serial smoke test for the Furuta pendulum hardware.

This script reads observation lines from the ESP32 firmware and repeatedly
sends "u 0" so the motor should remain stopped.  Use this before connecting
the PPO policy to verify serial parsing, timing, and observation values.

Expected ESP32 line format:
    obs=[cos_theta,sin_theta,theta_dot,phi,phi_dot] ...
"""

from __future__ import annotations

import argparse
import re
import sys
import time

try:
    import serial
except ImportError:  # pragma: no cover - user environment helper
    print("Missing dependency: pyserial")
    print("Install it with: python -m pip install pyserial")
    sys.exit(1)


OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def parse_obs(line: str) -> list[float] | None:
    match = OBS_RE.search(line)
    if not match:
        return None

    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) != 5:
        return None

    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def send_stop(port: serial.Serial) -> None:
    port.write(b"u 0\n")
    port.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read ESP32 observations and send u 0.")
    parser.add_argument("--port", default="COM5", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--seconds", type=float, default=20.0, help="How long to run")
    parser.add_argument("--stop-hz", type=float, default=5.0, help="How often to send u 0")
    args = parser.parse_args()

    print(f"Opening {args.port} at {args.baud} baud")
    print("This smoke test sends u 0 repeatedly. Motor should stay stopped.")
    print("Press Ctrl+C to stop.")

    obs_count = 0
    bad_count = 0
    start_time = time.perf_counter()
    last_obs_time = start_time
    last_stop_time = 0.0
    last_print_time = start_time

    try:
        with serial.Serial(args.port, args.baud, timeout=0.2) as port:
            time.sleep(2.0)
            port.reset_input_buffer()
            send_stop(port)

            while time.perf_counter() - start_time < args.seconds:
                now = time.perf_counter()

                if now - last_stop_time >= 1.0 / args.stop_hz:
                    send_stop(port)
                    last_stop_time = now

                raw = port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                obs = parse_obs(line)
                if obs is None:
                    if line and not line.startswith("#"):
                        bad_count += 1
                    continue

                obs_count += 1
                dt = now - last_obs_time
                last_obs_time = now

                if now - last_print_time >= 0.5:
                    cos_t, sin_t, theta_dot, phi, phi_dot = obs
                    rate = obs_count / max(now - start_time, 1e-9)
                    print(
                        "obs "
                        f"rate={rate:5.1f}Hz dt={dt * 1000:6.1f}ms "
                        f"cos={cos_t:+.3f} sin={sin_t:+.3f} "
                        f"theta_dot={theta_dot:+.3f} "
                        f"phi={phi:+.3f} phi_dot={phi_dot:+.3f}"
                    )
                    last_print_time = now

            send_stop(port)

    except KeyboardInterrupt:
        print("\nInterrupted. Sending stop if port is still available.")
        try:
            if "port" in locals() and port.is_open:
                send_stop(port)
        except Exception:
            pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        print("Check that Arduino Serial Monitor is closed and the ESP32 is on the selected COM port.")
        return 1

    elapsed = time.perf_counter() - start_time
    print(f"Done. Parsed {obs_count} observations in {elapsed:.1f}s.")
    if bad_count:
        print(f"Ignored {bad_count} non-observation lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
