"""
run.py — the swing-up -> handoff -> balance state machine.

Ties swingup.py and balance.py into one controller, exactly like Ben Katz's
single control() function that switches between the energy pump and the linear
balancer. This same logic is the operating loop on hardware later; only the
source of x (sim vs ESP32 obs) and the sink for a (sim vs motor command) change.

States:
    SWINGUP  -> energy-pump; when |theta| < handoff, switch to BALANCE
    BALANCE  -> LQR; if |theta| > fall, drop back to SWINGUP
"""
from __future__ import annotations

import numpy as np

import plant
from swingup import EnergySwingUp, _wrap
from balance import LQRBalance

SWINGUP, BALANCE = "SWINGUP", "BALANCE"


class SwingUpBalance:
    """Outputs an arm VELOCITY command. Swing-up sets it bang-bang; balance
    integrates its LQR acceleration into it (the velocity-source architecture)."""

    def __init__(self, handoff_deg: float = 15.0, handoff_thetadot: float = 4.0,
                 handoff_phi_deg: float = 30.0,
                 fall_deg: float = 30.0, v_max: float = 18.0,
                 swingup_v_max: float = 45.0, swingup_phi_max_deg: float = 80.0,
                 swingup_return_v: float = 8.0, balance_enabled: bool = True,
                 swingup: EnergySwingUp | None = None, balance: LQRBalance | None = None,
                 dt: float = plant.DT):
        self.balance_enabled = bool(balance_enabled)   # False -> swing-up only (no handoff)
        self.swingup = swingup or EnergySwingUp()
        self.balance = balance or LQRBalance()
        self.handoff = np.deg2rad(handoff_deg)
        self.handoff_thd = float(handoff_thetadot)   # only hand off a SLOW (catchable) rod
        self.handoff_phi = np.deg2rad(handoff_phi_deg)  # ...AND with the arm near center
        self.fall = np.deg2rad(fall_deg)
        self.balance_v_max = 7.0                     # cap arm speed while balancing (anti-rail)
        self.v_max = float(v_max)
        self.swingup_v_max = float(swingup_v_max)        # arm-speed cap during swing-up
        self.swingup_phi_max = np.deg2rad(swingup_phi_max_deg)  # HARD arm-position clamp
        self.swingup_return_v = float(swingup_return_v)  # speed to return when clamped
        self.swingup_soft_phi = np.deg2rad(125.0)        # start decelerating the arm here (soft wall)
        self.dt = float(dt)
        # Bootstrap: when the pendulum is near the bottom and still, the energy pump
        # has no coherent theta_dot to lock onto, so oscillate the arm OPEN-LOOP at the
        # pendulum's natural period (resonant forcing) to build the first swing.
        self.t = 0.0
        self.T_pend = 2.0 * np.pi / np.sqrt(abs(plant.ALPHA))   # ~0.45 s small-osc period
        self.boot_amp = np.deg2rad(35.0)    # bootstrap while swing from bottom < this
        self.boot_A = np.deg2rad(68.0)       # arm oscillation amplitude during bootstrap
        self.boot_kp = 20.0                  # arm position-servo gain during bootstrap
        self.state = SWINGUP
        self.n_catches = 0          # how many times it transitioned into BALANCE
        self.v_cmd = 0.0
        self.booted = False         # latch: bootstrap runs ONCE, then pure energy pump

    def reset(self):
        self.swingup.reset()
        self.balance.reset()
        self.state = SWINGUP
        self.n_catches = 0
        self.v_cmd = 0.0
        self.t = 0.0
        self.booted = False

    def __call__(self, x: np.ndarray) -> float:
        self.t += self.dt
        theta = abs(_wrap(float(x[plant.THETA])))
        phi = float(x[plant.PHI])
        if self.state == SWINGUP:
            amp = np.pi - theta                           # swing amplitude from the bottom
            theta_dot_now = float(x[plant.THETA_DOT])
            # Hand to the (phase-robust) energy pump as soon as the rod is moving at all;
            # only bootstrap a TRULY still rod (the open-loop forcing is phase-arbitrary).
            if amp >= self.boot_amp or abs(theta_dot_now) > 1.0:
                self.booted = True
            if not self.booted:
                # BOOTSTRAP (once): open-loop resonant forcing to wake a still pendulum.
                phi_target = self.boot_A * np.sign(np.sin(2.0 * np.pi * self.t / self.T_pend))
                self.v_cmd = self.boot_kp * (phi_target - phi)
            else:
                # energy-regulated proportional pump (returns a velocity directly)
                self.v_cmd = self.swingup(x)
            self.v_cmd = float(np.clip(self.v_cmd, -self.swingup_v_max, self.swingup_v_max))
            # SOFT WALL: ramp arm speed to 0 between soft_phi and 180 deg, so a fast arm
            # decelerates and can't coast past the +-180 limit (centering can't do this).
            pa = abs(phi)
            if pa > self.swingup_soft_phi and self.v_cmd * phi > 0.0:   # moving outward in the zone
                frac = max(0.0, (np.pi - pa) / (np.pi - self.swingup_soft_phi))
                self.v_cmd *= frac
            # FULL ROTATION: no arm clamp; arm can spin freely. Hand off a SLOW rod near
            # upright (arm position no longer matters since there's no cable limit).
            theta_dot = float(x[plant.THETA_DOT])
            if theta < self.handoff and abs(theta_dot) < self.handoff_thd and self.balance_enabled:
                self.state = BALANCE
                self.n_catches += 1
                self.v_cmd = float(x[plant.PHI_DOT])      # start balance from the REAL arm velocity
                self.balance.reset()                      # re-init the observer for this catch
        else:  # BALANCE
            if theta > np.deg2rad(45.0):
                # rod fell out of the linear region: linear LQR is invalid here -> don't
                # wind the integrator and spin the arm. Bleed v_cmd to zero (or re-pump).
                self.v_cmd *= 0.5
                if theta > self.fall:
                    self.state = SWINGUP                  # let swing-up bring it back up
                    self.booted = True
            else:
                self.v_cmd += self.balance(x) * self.dt   # integrate LQR accel -> velocity
                self.v_cmd = float(np.clip(self.v_cmd, -self.balance_v_max, self.balance_v_max))
        return self.v_cmd


# ------------------------------------------------------------------
def simulate(seconds: float = 8.0, start_hanging: bool = True, actuator=None):
    from sim import FurutaSim, ArmActuator
    if actuator is None:
        actuator = ArmActuator(ideal=True)     # default: perfect velocity source
    sim = FurutaSim(phi_dot_max=150.0)
    ctrl = SwingUpBalance()
    ctrl.reset()

    theta0 = (np.pi - np.deg2rad(2.0)) if start_hanging else np.deg2rad(0.0)
    x = sim.reset(theta0=theta0)

    n = int(seconds / sim.dt)
    log = {"t": [], "theta": [], "phi": [], "state": [], "a": []}
    first_catch_t = None
    balance_steps = 0
    max_phi = 0.0
    for k in range(n):
        v = ctrl(x)                                       # arm velocity command
        a = actuator.realize(v, float(x[plant.PHI_DOT]))  # -> realized arm accel
        x = sim.step(a)
        t = k * sim.dt
        th = _wrap(float(x[plant.THETA]))
        log["t"].append(t); log["theta"].append(np.degrees(th))
        log["phi"].append(np.degrees(float(x[plant.PHI])))
        log["state"].append(ctrl.state); log["a"].append(a)
        if first_catch_t is None and ctrl.state == BALANCE:
            first_catch_t = t
        if ctrl.state == BALANCE:
            balance_steps += 1
        max_phi = max(max_phi, abs(float(x[plant.PHI])))
    return ctrl, sim, log, dict(first_catch_t=first_catch_t,
                                balance_steps=balance_steps,
                                max_phi=max_phi)


def _report(label, seconds, ctrl, log, stats):
    last2 = [abs(th) for th, t in zip(log["theta"], log["t"]) if t >= seconds - 2.0]
    held = max(last2) < 12.0 if last2 else False
    balance_frac = stats["balance_steps"] / len(log["t"])
    fc = stats["first_catch_t"]
    # limit-cycle amplitude: spread of theta in the last 2 s
    lc = (max(last2) if last2 else 0.0)
    print(f"\n===== {label} =====")
    print(f"first catch (handoff)   : {fc:.2f} s" if fc is not None else "first catch: NEVER")
    print(f"time balancing          : {100*balance_frac:.0f}%   re-pumps: {ctrl.n_catches}")
    print(f"max arm excursion       : {np.degrees(stats['max_phi']):.0f} deg (limit ~120)")
    print(f"last-2s |theta| max     : {lc:.1f} deg  (jitter/limit-cycle near upright)")
    print(f"VERDICT                 : {'HELD UPRIGHT' if held else 'DID NOT HOLD'}")
    return held


if __name__ == "__main__":
    from sim import ArmActuator
    seconds = 16.0

    # 1) ideal actuator (perfectly smooth arm velocity) -- the original design check
    c1, _, log1, s1 = simulate(seconds=seconds, actuator=None)
    _report("IDEAL actuator (smooth velocity)", seconds, c1, log1, s1)

    # 2) real actuator: 1.5 rad/s deadzone + 3 rad/s min sustained speed (measured)
    c2, _, log2, s2 = simulate(seconds=seconds, actuator=ArmActuator())
    held2 = _report("REAL actuator (deadzone + 3 rad/s floor)", seconds, c2, log2, s2)

    print("\n" + ("=> Balance SURVIVES the real actuator (proceed to hw_env.py)."
                  if held2 else
                  "=> Balance BREAKS on the real actuator -- adjust before hardware."))
