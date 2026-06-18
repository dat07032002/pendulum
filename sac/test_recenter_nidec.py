"""
Standalone recenter test for the Nidec (speed-controlled) arm.

Displaces the arm to +/-40 deg programmatically, then runs a proportional
velocity servo (u = -Kp*phi, clamped) to bring it back to center, logging the
phi trajectory so we can confirm it servos in smoothly without overshooting
the limits.

This IS the recenter algorithm that will be ported into furuta_hw_env.py once
verified. Gentle gains for the first test.

Safety: speeds capped, script aborts at +/-110 deg, firmware backstop at 120,
"s" sent on every exit. The motor is driven only via short u commands (the
firmware watchdog also stops it if this script dies).
"""

import math
import re
import time

import serial

PORT = "COM5"

# --- recenter servo params (gentle first pass) ---
KP = 0.5            # u per radian of error
U_MAX = 0.30        # recenter speed cap (|u|)
TOL_DEG = 8.0       # "centered" tolerance
SETTLE_S = 0.3      # must stay within tol this long
ABORT_DEG = 110.0   # script-side safety abort

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def send(p, cmd):
    p.write((cmd + "\n").encode("ascii"))
    p.flush()


def latest_phi(p):
    """Drain all pending serial, return the freshest phi (rad) or None."""
    phi = None
    while p.in_waiting:
        line = p.readline().decode("utf-8", errors="replace").strip()
        m = OBS_RE.search(line)
        if m:
            try:
                phi = float(m.group(1).split(",")[3])
            except (ValueError, IndexError):
                pass
    return phi


def phi_blocking(p, timeout=0.5):
    end = time.time() + timeout
    val = None
    while time.time() < end:
        v = latest_phi(p)
        if v is not None:
            val = v
        time.sleep(0.005)
    return val


def displace(p, target_deg, u):
    """Drive arm to ~target_deg, then stop and settle."""
    print(f"  displacing toward {target_deg:+.0f} deg at u={u:+.2f} ...")
    end = time.time() + 4.0
    while time.time() < end:
        send(p, f"u {u:.3f}")
        phi = latest_phi(p)
        if phi is not None and abs(math.degrees(phi)) >= abs(target_deg):
            break
        time.sleep(0.015)
    send(p, "s")
    time.sleep(0.8)


def recenter(p):
    """P-servo phi -> 0. Returns the phi_deg trajectory."""
    traj = []
    settled_since = None
    last_print = 0.0
    start = time.time()
    while time.time() - start < 8.0:
        phi = latest_phi(p)
        if phi is None:
            time.sleep(0.005)
            continue
        phi_deg = math.degrees(phi)
        traj.append(phi_deg)

        if abs(phi_deg) > ABORT_DEG:
            send(p, "s")
            print(f"  ABORT: |phi|={phi_deg:.0f} > {ABORT_DEG}")
            break

        if abs(phi_deg) <= TOL_DEG:
            if settled_since is None:
                settled_since = time.time()
            elif time.time() - settled_since >= SETTLE_S:
                send(p, "s")
                break
        else:
            settled_since = None

        u = max(-U_MAX, min(U_MAX, -KP * phi))   # drive toward center
        send(p, f"u {u:.3f}")

        now = time.time()
        if now - last_print >= 0.25:
            print(f"    phi={phi_deg:+6.1f} deg  u={u:+.3f}")
            last_print = now
        time.sleep(0.015)
    send(p, "s")
    return traj


def main():
    p = serial.Serial(PORT, 115200, timeout=0.02)
    time.sleep(2.0)
    p.reset_input_buffer()
    send(p, "s")
    time.sleep(0.3)

    send(p, "z")          # define current arm position as center (phi = 0)
    time.sleep(0.2)
    print("zeroed: center = current arm position.\n")

    try:
        for target in (+40, -40):
            print(f"=== displace {target:+d} deg, then recenter ===")
            displace(p, target, 0.4 if target > 0 else -0.4)
            phi0 = phi_blocking(p, 0.4)
            print(f"  displaced to phi={math.degrees(phi0):+.1f} deg")
            traj = recenter(p)
            phif = phi_blocking(p, 0.4)
            if traj:
                lo, hi = min(traj), max(traj)
                print(f"  result: start {traj[0]:+.1f} -> end {math.degrees(phif):+.1f} deg")
                print(f"          trajectory range [{lo:+.1f}, {hi:+.1f}] deg")
                ok = abs(math.degrees(phif)) <= TOL_DEG + 5
                print(f"  -> {'OK' if ok else 'CHECK'}: settled within tolerance\n")
            time.sleep(0.6)
    finally:
        for _ in range(6):
            send(p, "s")
            time.sleep(0.05)
        p.close()
        print("done. motor stopped.")


if __name__ == "__main__":
    main()
