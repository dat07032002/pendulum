"""
Energy-pumping swing-up test (standalone) for the Nidec Furuta.

Pumps energy into the pendulum until it reaches near upright, then stops
(this is the hand-off point to a balance controller).

Control law:
  E = 0.5*theta_dot^2 + OMEGA2*(cos(theta) - 1)      # 0 at upright, negative below
  u = clamp( PUMP_SIGN * K * (-E) * sign(theta_dot*cos(theta)), -U_MAX, U_MAX )

theta = 0 is upright, +/-pi hanging (from cos/sin in the obs).

SAFETY: drives the pendulum vigorously. 12V on, HAND ON POWER, area clear.
Bounded by U_MAX, the firmware phi backstop, and a hard TIMEOUT. Sends 's'
on every exit. If it de-energizes (pendulum won't rise), flip PUMP_SIGN.
"""
import math
import re
import time
import serial

PORT = "COM5"
OMEGA2 = 80.0        # ~ g/L; affects pump scaling (tune)
K = 0.5              # pump gain
U_MAX = 0.30         # arm-speed cap (modest for the fast motor)
PUMP_SIGN = 1        # flip to -1 if it loses energy instead of gaining
UPRIGHT_DEG = 25.0   # stop / hand-off when within this of upright
TIMEOUT = 15.0       # hard stop

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def main():
    try:
        p = serial.Serial(PORT, 115200, timeout=0.02)
    except Exception as e:
        print(f"COM5 busy/unavailable ({e}). Stop training first.")
        return
    time.sleep(2.0)
    p.reset_input_buffer()

    def send(c):
        p.write((c + "\n").encode()); p.flush()

    def latest():
        cos = sin = thd = None
        while p.in_waiting:
            m = OBS_RE.search(p.readline().decode("utf-8", "replace").strip())
            if m:
                try:
                    v = m.group(1).split(",")
                    cos, sin, thd = float(v[0]), float(v[1]), float(v[2])
                except (ValueError, IndexError):
                    pass
        return cos, sin, thd

    send("s"); time.sleep(0.3)
    send("z"); time.sleep(0.2)
    print(f"swing-up: OMEGA2={OMEGA2} K={K} U_MAX={U_MAX} PUMP_SIGN={PUMP_SIGN}")
    print(f"HAND ON POWER. Pumping until |theta|<{UPRIGHT_DEG}deg or {TIMEOUT}s...")

    start = time.time(); last = 0.0; success = False; closest = 180.0
    try:
        while time.time() - start < TIMEOUT:
            cos, sin, thd = latest()
            if cos is None:
                time.sleep(0.005); continue
            theta = math.atan2(sin, cos)          # 0 = upright
            deg = math.degrees(theta)
            closest = min(closest, abs(deg))
            if abs(deg) < UPRIGHT_DEG:
                send("s"); success = True
                print(f"  >>> reached upright! theta={deg:+.1f}deg (hand-off point)")
                break
            E = 0.5 * thd * thd + OMEGA2 * (cos - 1.0)
            direction = 1.0 if (thd * cos) > 0 else -1.0
            u = PUMP_SIGN * K * (-E) * direction
            u = max(-U_MAX, min(U_MAX, u))
            send(f"u {u:.3f}")
            now = time.time() - start
            if now - last >= 0.3:
                print(f"  t={now:4.1f}s theta={deg:+7.1f} thdot={thd:+6.2f} E={E:+7.1f} u={u:+.2f}")
                last = now
            time.sleep(0.02)
    finally:
        for _ in range(6):
            send("s"); time.sleep(0.05)
        p.close()
    print(f"\n{'SUCCESS' if success else 'did NOT reach upright'} | closest to upright: {closest:.0f}deg")
    if not success:
        print("If theta stayed near +/-180 (no progress), flip PUMP_SIGN to", -PUMP_SIGN)


if __name__ == "__main__":
    main()
