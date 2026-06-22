"""run_ftest.py — send 'ftest' (short-pulse + guaranteed-brake fine-speed test) and capture
the table. MOVES THE ARM (sweeps; firmware aborts past 100 deg)."""
from __future__ import annotations
import time
import serial
import config

def main():
    ser = serial.Serial(config.PORT, config.BAUD, timeout=1.0)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.write(b"ftest\n"); ser.flush()
    print("sent ftest; collecting (up to ~90s)...\n")
    deadline = time.time() + 120
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if line.startswith("everyN=") or line.startswith("# ftest") or "ABORT" in line:
            print(line)
        if line == "DONE":
            print("\n[DONE]")
            break
    ser.write(b"s\n"); ser.flush()
    ser.close()

if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nClose balance_chip.py / Serial Monitor first (COM5 busy).")
