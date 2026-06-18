"""
Quick encoder scaling check (motor off, read-only).

Zeros the arm encoder, then live-prints phi_deg while you rotate the arm by
hand. Turn the arm exactly ONE FULL REVOLUTION and read phi_deg:
  - reads ~360 (or ~-360)  -> CPR is correct
  - reads ~720             -> CPR still half (firmware not reflashed with 400)
  - reads ~180             -> CPR is double the truth

Run with COM5 free (stop any training first). Ctrl+C to stop.
"""
import re
import time
import serial

PHI_RE = re.compile(r"phi_deg=([-\d.]+)")

p = serial.Serial("COM5", 115200, timeout=0.1)
time.sleep(2.0)              # ESP32 reboots on port open
p.reset_input_buffer()
p.write(b"z\n"); p.flush()   # zero the arm encoder
time.sleep(0.3)

print("Encoder zeroed. Rotate the arm ONE FULL TURN slowly, then read phi_deg.")
print("Expect ~360 (or ~-360) for one full revolution. Ctrl+C to stop.\n")

try:
    while True:
        line = p.readline().decode("utf-8", errors="replace").strip()
        m = PHI_RE.search(line)
        if m:
            print(f"\r  phi = {float(m.group(1)):+8.1f} deg    ", end="", flush=True)
except KeyboardInterrupt:
    pass
finally:
    p.write(b"s\n"); p.flush(); p.close()
    print("\nstopped.")
