"""
Send one short normalized motor command to the ESP32 and read observations.

The ESP32 firmware should accept commands like:
    u 0.05
    u 0

This script always sends "u 0" before exiting.
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


def send_u(port: serial.Serial, u: float) -> None:
    port.write(f"u {u:.5f}\n".encode("ascii"))
    port.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one short u command to ESP32.")
    parser.add_argument("--port", default="COM5", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--u", type=float, default=0.05, help="Normalized command [-1, 1]")
    parser.add_argument("--duration", type=float, default=0.3, help="Command duration in seconds")
    parser.add_argument("--pre-stop", type=float, default=0.5, help="Seconds to send u 0 before pulse")
    parser.add_argument("--post-read", type=float, default=1.0, help="Seconds to keep reading after stop")
    args = parser.parse_args()

    if not -1.0 <= args.u <= 1.0:
        print("--u must be in [-1, 1]")
        return 1

    print(f"Opening {args.port} at {args.baud} baud")
    print(f"Pulse: u={args.u:+.3f} for {args.duration:.3f}s, then u 0")
    print("Keep hand near power switch. Press Ctrl+C to stop.")

    start_time = time.perf_counter()
    pulse_start = None
    stopped = False
    last_print_time = 0.0
    obs_count = 0

    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as port:
            time.sleep(2.0)
            port.reset_input_buffer()

            send_u(port, 0.0)
            command_start = time.perf_counter() + args.pre_stop
            end_time = command_start + args.duration + args.post_read

            while time.perf_counter() < end_time:
                now = time.perf_counter()

                if pulse_start is None and now >= command_start:
                    send_u(port, args.u)
                    pulse_start = now
                    print(f"Sent u {args.u:+.3f}")

                if pulse_start is not None and not stopped and now - pulse_start >= args.duration:
                    send_u(port, 0.0)
                    stopped = True
                    print("Sent u 0")

                raw = port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                obs = parse_obs(line)
                if obs is None:
                    continue

                obs_count += 1
                if now - last_print_time >= 0.1:
                    cos_t, sin_t, theta_dot, phi, phi_dot = obs
                    elapsed = now - start_time
                    print(
                        f"t={elapsed:5.2f}s "
                        f"cos={cos_t:+.3f} sin={sin_t:+.3f} "
                        f"theta_dot={theta_dot:+.3f} "
                        f"phi={phi:+.3f} phi_dot={phi_dot:+.3f}"
                    )
                    last_print_time = now

            send_u(port, 0.0)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        try:
            if "port" in locals() and port.is_open:
                send_u(port, 0.0)
                print("Sent u 0")
        except Exception:
            pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        print("Close Arduino Serial Monitor and check the selected COM port.")
        return 1

    print(f"Done. Parsed {obs_count} observations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
