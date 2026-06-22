"""
hw_env.py — run the swing-up/balance controller on the real Furuta rig.
** MOVES THE MOTOR / PENDULUM. ** Requires nidec_velocity.ino flashed.

Mirrors run.py exactly: read obs from the ESP32, build x = [phi, theta, phi_dot,
theta_dot], run the SwingUpBalance controller (which integrates acceleration into a
velocity command v_cmd), and send "v <v_cmd>" to the firmware. The firmware owns the
velocity->duty map, break-free kick, deadzone, and the phi backstop.

Modes:
  lift  : motor off; you lift the pendulum near upright, the LQR catches and holds it
          (safest first test -- balance only, no swing-up).
  full  : energy swing-up -> handoff -> balance (the whole thing).

Safety: sends "v 0" then "s" on exit / Ctrl-C; stops on obs timeout; the firmware
also has a 120 deg phi backstop and a watchdog.

Usage: python hw_env.py [--mode lift|full] [--port COM5] [--seconds 30]
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import serial

import config
import plant
from run import SwingUpBalance, SWINGUP, BALANCE
from balance import _wrap

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def parse_obs(line: str):
    m = OBS_RE.search(line)
    if not m:
        return None
    p = m.group(1).split(",")
    if len(p) != 5:
        return None
    try:
        return [float(x) for x in p]
    except ValueError:
        return None


def obs_to_state(o) -> np.ndarray:
    """obs=[cos_theta, sin_theta, theta_dot, phi, phi_dot] -> x=[phi,theta,phi_dot,theta_dot]."""
    cos_t, sin_t, theta_dot, phi, phi_dot = o
    theta = float(np.arctan2(sin_t, cos_t))     # 0 = upright
    return np.array([phi, theta, phi_dot, theta_dot], dtype=float)


class Link:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.02)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.send_v(0.0)

    def send_v(self, v):
        self.ser.write(f"v {v:.4f}\n".encode("ascii")); self.ser.flush()

    def read_obs(self, timeout=0.05):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            o = parse_obs(raw.decode("utf-8", errors="replace"))
            if o is not None:
                return o
        return None

    def stop(self):
        try:
            self.send_v(0.0); time.sleep(0.05)
            self.ser.write(b"s\n"); self.ser.flush()
        finally:
            self.ser.close()


def wait_for_lift(link, handoff_deg=12.0, settle=0.1):
    """Motor off; wait until the user holds the pendulum near upright and still."""
    print("\a>>> Lift the pendulum to upright and hold steady; the LQR will catch it...")
    handoff = np.deg2rad(handoff_deg)
    settled_since = None
    while True:
        link.send_v(0.0)
        o = link.read_obs(timeout=0.2)
        if o is None:
            continue
        x = obs_to_state(o)
        if abs(_wrap(x[plant.THETA])) < handoff and abs(x[plant.THETA_DOT]) < 2.0:
            settled_since = settled_since or time.perf_counter()
            if time.perf_counter() - settled_since >= settle:
                print("\a>>> caught -> LQR active, let go")
                return x
        else:
            settled_since = None


def main():
    ap = argparse.ArgumentParser(description="Run swing-up/balance on hardware (MOVES THE RIG).")
    ap.add_argument("--mode", choices=["lift", "full", "swingup"], default="swingup")
    ap.add_argument("--port", default=config.PORT)
    ap.add_argument("--seconds", type=float, default=60.0)
    # swing-up tuning (also adjustable LIVE via stdin: e.g. "ke 8", "coast 0.04", "vmax 40")
    ap.add_argument("--ke", type=float, default=6.0, help="energy-pump gain")
    ap.add_argument("--coast", type=float, default=0.06, help="coast/handoff energy band")
    ap.add_argument("--vmax", type=float, default=45.0, help="swing-up arm-speed ceiling [rad/s]")
    ap.add_argument("--handoff-thd", type=float, default=4.0, help="max |theta_dot| to hand off [rad/s]")
    ap.add_argument("--kcenter", type=float, default=2.0, help="arm-centering gain (bounds arm to +-180)")
    ap.add_argument("--soft", type=float, default=125.0, help="soft-wall start angle [deg] (lower = tighter to 180)")
    ap.add_argument("--R", type=float, default=3.0, help="balance LQR R (higher = gentler)")
    ap.add_argument("--bvmax", type=float, default=7.0, help="balance arm-speed cap [rad/s] (anti-rail)")
    args = ap.parse_args()

    if not config.is_calibrated():
        print("NOTE: config not calibrated, but the firmware owns the velocity map, so that's OK.")

    from swingup import EnergySwingUp
    from balance import LQRBalance
    su = EnergySwingUp(k_e=args.ke, v_max=args.vmax, coast_band=args.coast, k_center=args.kcenter)
    bal = LQRBalance(R=np.array([[args.R]]))
    ctrl = SwingUpBalance(swingup=su, balance=bal, swingup_v_max=args.vmax,
                          handoff_thetadot=args.handoff_thd,
                          balance_enabled=(args.mode != "swingup"))
    ctrl.swingup_soft_phi = np.deg2rad(args.soft)
    ctrl.balance_v_max = args.bvmax
    print(f"balance K = {np.round(bal.K.ravel(), 2)}  (R={args.R}, arm cap={args.bvmax} rad/s)")
    ctrl.reset()

    # ---- live tuning: background thread reads stdin and updates params on the fly ----
    import threading
    def tuner():
        try:
            stdin_iter = iter(sys.stdin)
        except Exception:
            return
        for line in stdin_iter:
            p = line.split()
            if not p:
                continue
            try:
                if p[0] == "ke":      su.k_e = float(p[1]);        print(f"  >> k_e={su.k_e}")
                elif p[0] == "coast": su.coast_band = float(p[1]); print(f"  >> coast={su.coast_band}")
                elif p[0] == "vmax":  su.v_max = ctrl.swingup_v_max = float(p[1]); print(f"  >> vmax={su.v_max}")
                elif p[0] == "thd":   ctrl.handoff_thd = float(p[1]); print(f"  >> handoff_thd={ctrl.handoff_thd}")
                elif p[0] in ("kc", "kcenter"): su.k_center = float(p[1]); print(f"  >> k_center={su.k_center}")
                elif p[0] == "soft": ctrl.swingup_soft_phi = np.deg2rad(float(p[1])); print(f"  >> soft_phi={p[1]}deg")
                # --- live LQR balance tuning (rebuilds K) ---
                elif p[0] == "R":     print("  >> K=", np.round(ctrl.balance.set_weights(R=float(p[1])).ravel(), 2))
                elif p[0] == "qth":   print("  >> K=", np.round(ctrl.balance.set_weights(q_theta=float(p[1])).ravel(), 2))
                elif p[0] == "qthd":  print("  >> K=", np.round(ctrl.balance.set_weights(q_thetadot=float(p[1])).ravel(), 2))
                elif p[0] == "qphi":  print("  >> K=", np.round(ctrl.balance.set_weights(q_phi=float(p[1])).ravel(), 2))
                elif p[0] == "bvmax": ctrl.balance_v_max = float(p[1]); print(f"  >> balance arm cap={ctrl.balance_v_max}")
                elif p[0] in ("q", "stop"): break
                else: print("  cmds: ke|coast|vmax|thd|kc|soft <v> ; balance: R|qth|qthd|qphi <v> ; q")
            except (IndexError, ValueError):
                print("  usage: ke 8")
    threading.Thread(target=tuner, daemon=True).start()
    print("LIVE TUNING: type e.g. 'ke 8' then Enter while it runs. 'q' to quit.")

    print(f"** hw_env MOVES THE RIG (mode={args.mode}). Keep clear. **")
    link = Link(args.port, config.BAUD)
    try:
        if args.mode == "lift":
            ctrl.state = BALANCE                 # skip swing-up; start in balance
            wait_for_lift(link)
        # full mode: ctrl starts in SWINGUP (default) and pumps up from hanging

        t0 = time.perf_counter()
        last_state = None
        frozen = 0
        prev_cs = None
        max_amp = 0.0           # max swing amplitude from the bottom [deg]
        best_arrival = 999.0    # slowest |theta_dot| seen while near the top (catchable)
        n_near_top = 0          # how many control steps were within 20 deg of upright
        last_print = t0
        while time.perf_counter() - t0 < args.seconds:
            o = link.read_obs(timeout=0.2)
            if o is None:
                link.send_v(0.0)
                print("Safety stop: obs timeout.")
                break
            x = obs_to_state(o)
            phi, theta = float(x[plant.PHI]), _wrap(float(x[plant.THETA]))
            theta_dot = float(x[plant.THETA_DOT])

            # frozen-theta guard (AS5600 cable): cos/sin exactly constant while driven
            cs = (o[0], o[1])
            if cs == prev_cs and abs(ctrl.v_cmd) > 1.0:
                frozen += 1
            else:
                frozen = 0
            prev_cs = cs
            if frozen > 200:    # ~1 s: a real disconnect freezes forever; tolerate EMI glitches
                link.send_v(0.0)
                print("Safety stop: theta sensor frozen (check AS5600).")
                break

            # FULL ROTATION: no arm-position safety stop (arm spins freely).
            v_cmd = ctrl(x)
            link.send_v(v_cmd)

            # telemetry: swing amplitude from the bottom (0 = hanging, 180 = upright)
            amp = 180.0 - abs(np.degrees(theta))
            max_amp = max(max_amp, amp)
            if abs(np.degrees(theta)) < 20.0:          # near the top: track arrival speed
                n_near_top += 1
                best_arrival = min(best_arrival, abs(theta_dot))
            now = time.perf_counter()
            if now - last_print >= 0.4:
                print(f"  [{now-t0:5.1f}s] {ctrl.state:7s} theta={np.degrees(theta):+5.0f} "
                      f"thd={theta_dot:+5.1f} amp={amp:5.1f}(max{max_amp:5.0f}) "
                      f"phi={np.degrees(phi):+5.0f} v={v_cmd:+5.1f} | slowest@top={best_arrival:4.1f}")
                last_print = now

            if ctrl.state != last_state:
                tag = "BALANCE (caught)" if ctrl.state == BALANCE else "SWINGUP (pumping)"
                print(f"  [{now-t0:5.1f}s] -> {tag}  theta={np.degrees(theta):+.0f}deg")
                last_state = ctrl.state
        print(f"\n=== swing-up summary ===  max amp={max_amp:.0f} deg   "
              f"slowest arrival near top={best_arrival:.1f} rad/s   "
              f"time near top={n_near_top*config.CONTROL_DT:.1f}s")
        print("(robust+slow = max amp ~180 AND slowest@top small, reached repeatedly)")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        link.stop()
        print("motor stopped.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nIs the ESP32 on {config.PORT}? Close other serial monitors.")
