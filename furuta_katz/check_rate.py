"""
Measure the ESP32 obs streaming rate and jitter — READ ONLY.

Opens the serial port, never writes a command (the motor stays disabled: the
firmware calls motorStop() in setup and the watchdog keeps it stopped while no
"u" is sent). Counts valid `obs=[...]` lines over a window and reports the
achieved rate, mean control period, and jitter.

Usage:
    python check_rate.py [--port COM5] [--seconds 5] [--baud 921600]
"""
from __future__ import annotations

import argparse
import re
import time

import serial
import serial.tools.list_ports

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def autodetect_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in desc for k in ("cp210", "ch340", "ch910", "silicon labs", "usb-serial", "uart")):
            return p.device
    return ports[0].device if ports else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure ESP32 obs rate/jitter (read-only).")
    ap.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    ap.add_argument("--seconds", type=float, default=5.0, help="measurement window")
    ap.add_argument("--baud", type=int, default=921600)
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if port is None:
        print("No serial ports found. Plug in the ESP32 or pass --port.")
        return
    print(f"Ports available: {[p.device for p in serial.tools.list_ports.comports()]}")
    print(f"Opening {port} @ {args.baud} (read-only, no commands sent)...")

    with serial.Serial(port, args.baud, timeout=0.2) as ser:
        time.sleep(2.0)              # ESP32 resets on serial open
        ser.reset_input_buffer()

        # Warm up: wait for the first valid obs so we don't time the boot banner.
        t_warm = time.perf_counter()
        while time.perf_counter() - t_warm < 3.0:
            raw = ser.readline()
            if raw and OBS_RE.search(raw.decode("utf-8", errors="replace")):
                break
        else:
            print("No obs=[...] lines seen in 3 s. Is nidec_policy.ino flashed?")
            return

        stamps: list[float] = []
        non_obs = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if OBS_RE.search(line):
                stamps.append(time.perf_counter())
            else:
                non_obs += 1

    n = len(stamps)
    if n < 2:
        print(f"Only {n} obs lines captured — link or firmware problem.")
        return

    span = stamps[-1] - stamps[0]
    rate = (n - 1) / span
    intervals = [(stamps[i + 1] - stamps[i]) * 1e3 for i in range(n - 1)]  # ms
    intervals_sorted = sorted(intervals)
    mean = sum(intervals) / len(intervals)
    var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    std = var ** 0.5
    p50 = intervals_sorted[len(intervals_sorted) // 2]
    p95 = intervals_sorted[int(0.95 * (len(intervals_sorted) - 1))]
    p99 = intervals_sorted[int(0.99 * (len(intervals_sorted) - 1))]

    print("\n===== ESP32 obs rate =====")
    print(f"obs lines           : {n} over {span:.2f} s")
    print(f"non-obs lines        : {non_obs} (banners / health / debug)")
    print(f"achieved rate        : {rate:6.1f} Hz")
    print(f"mean period          : {mean:6.2f} ms")
    print(f"jitter (std)         : {std:6.2f} ms")
    print(f"period p50 / p95 / p99: {p50:.2f} / {p95:.2f} / {p99:.2f} ms")
    print(f"min / max period     : {min(intervals):.2f} / {max(intervals):.2f} ms")
    print("==========================")
    target = 5.0
    print(f"\nFor a solid 200 Hz you want mean ~{target:.0f} ms with low p99.")
    if rate >= 190:
        print("=> At/near 200 Hz already.")
    elif rate >= 140:
        print(f"=> ~{rate:.0f} Hz. Below 200; the fixed delay(5)+work is the cause.")
    else:
        print(f"=> {rate:.0f} Hz — well under 200; firmware loop needs rework for 200 Hz.")


if __name__ == "__main__":
    main()
