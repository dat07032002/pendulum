"""
Grab test: is the Nidec open-loop voltage (torque-ish) or closed-loop speed?

The arm spins gently back and forth at u=0.15. RESIST it with your hand and
watch phi_dot:
  - phi_dot HOLDS its value under your hand  -> closed-loop SPEED control
       (motor fights to keep RPM -> bad for balancing, needs a torque driver)
  - phi_dot DROPS toward 0 under your hand   -> voltage / torque control
       (motor yields -> GOOD, the torque-trained policy will deploy)

Motor only (no pendulum needed). Hand on power. Ctrl+C stops + sends 's'.
"""
import math
import re
import time
import serial

OBS = re.compile(r"obs=\[([^\]]+)\]")


def open_port():
    for baud in (921600, 115200):
        try:
            p = serial.Serial("COM5", baud, timeout=0.05)
            time.sleep(2.0)
            t0 = time.time()
            while time.time() - t0 < 2.0:
                if "obs=" in p.readline().decode("utf-8", "replace"):
                    print(f"connected at {baud} baud")
                    p.reset_input_buffer()
                    return p
            p.close()
        except Exception:
            pass
    raise SystemExit("No obs on COM5 at 921600 or 115200 -- is the ESP32 plugged in / port free?")


def main():
    p = open_port()

    def send(c):
        p.write((c + "\n").encode()); p.flush()

    def phidot():
        v = None
        while p.in_waiting:
            m = OBS.search(p.readline().decode("utf-8", "replace"))
            if m:
                try:
                    v = float(m.group(1).split(",")[4])
                except (ValueError, IndexError):
                    pass
        return v

    print("\nGRAB TEST -- arm spins back and forth. RESIST it with your hand, watch phi_dot:")
    print("  HOLDS under your hand -> closed-loop SPEED control (needs torque driver)")
    print("  DROPS under your hand -> voltage/torque control (GOOD)\n")
    try:
        d = 0.15
        t_flip = time.time()
        while True:
            if time.time() - t_flip > 2.0:
                d = -d
                t_flip = time.time()
            send(f"u {d:.3f}")
            pv = phidot()
            if pv is not None:
                bar = "#" * int(min(abs(pv) * 3, 40))
                print(f"\r u={d:+.2f}  phi_dot={pv:+6.2f} rad/s ({math.degrees(pv):+7.0f} deg/s) {bar}      ",
                      end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(6):
            send("s"); time.sleep(0.05)
        p.close()
        print("\nstopped.")


if __name__ == "__main__":
    main()
