"""
actuator_ceiling.py — what is the BEST achievable balance on the real Nidec actuator?

Runs the firmware's exact LQR-with-integrator balance (a=-Kx, vcmd += a*dt, then the
motor quantizes vcmd) against three actuator models, from small catches, and reports the
longest hold. This tells us whether the deadzone+floor is the wall, and whether perfect
sub-floor fine control (dither) would rescue it -- the cheap answer before investing in RL.

  IDEAL        : perfect velocity source (control upper bound)
  REAL         : measured Nidec -- 1.5 rad/s deadzone, 3 rad/s min sustained
  DITHER-IDEAL : optimistic fine control -- 0.3 deadzone, 0.5 rad/s floor (best dither could do)

NOTE: stiction and pulse-momentum are NOT modeled; they would make REAL *worse*, not better.
So REAL here is an optimistic upper bound on the real motor.
"""
from __future__ import annotations

import numpy as np

import plant
from sim import FurutaSim, ArmActuator
from balance import _wrap

K_FIRMWARE = np.array([-18.31577, 855.90381, -9.14419, 61.18292])  # on-chip LQR default


def run_catch(actuator, K, tilt_deg, bvmax, seconds=6.0, a_max=400.0):
    """Hold time [s] from a tilt, using the firmware's integrate-accel-to-velocity LQR."""
    sim = FurutaSim(phi_dot_max=actuator.v_max)
    x = sim.reset(theta0=np.deg2rad(tilt_deg))
    dt = sim.dt
    vcmd = 0.0
    best = streak = 0
    for _ in range(int(seconds / dt)):
        xv = np.array([x[plant.PHI], _wrap(x[plant.THETA]), x[plant.PHI_DOT], x[plant.THETA_DOT]])
        a = float(np.clip(-(K @ xv), -a_max, a_max))
        vcmd = float(np.clip(vcmd + a * dt, -bvmax, bvmax))
        a_real = actuator.realize(vcmd, float(x[plant.PHI_DOT]))
        x = sim.step(a_real)
        upright = abs(_wrap(float(x[plant.THETA]))) < np.deg2rad(15.0)
        in_range = abs(float(x[plant.PHI])) < np.deg2rad(175.0)
        if upright and in_range:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
            if abs(_wrap(float(x[plant.THETA]))) > np.deg2rad(60.0):
                break
    return best * dt


def best_over_sweep(actuator, K, tilt_deg):
    """Best hold across a bvmax sweep (the cap mattered a lot on hardware)."""
    return max(run_catch(actuator, K, tilt_deg, bv) for bv in (6, 9, 15, 30))


def main():
    actuators = {
        "IDEAL        (perfect velocity)":      ArmActuator(ideal=True),
        "REAL         (deadzone 1.5, floor 3)": ArmActuator(v_deadzone=1.5, v_min=2.98),
        "DITHER-IDEAL (deadzone 0.3, floor .5)": ArmActuator(v_deadzone=0.3, v_min=0.5),
    }
    tilts = (2, 5, 10)
    print("Best hold time [s] (best over bvmax sweep), firmware LQR, 6 s window:\n")
    print(f"{'actuator':<38} " + "  ".join(f"{t}deg" for t in tilts))
    print("-" * 60)
    for name, act in actuators.items():
        holds = [best_over_sweep(act, K_FIRMWARE, t) for t in tilts]
        print(f"{name:<38} " + "  ".join(f"{h:4.2f}" for h in holds))
    print("\n(6.00 = held the whole window. <1 = falls almost immediately.)")
    print("If REAL collapses but DITHER-IDEAL holds -> fine control is the fix (dither/stepper).")
    print("If DITHER-IDEAL also collapses -> even perfect fine control can't, deeper issue.")


if __name__ == "__main__":
    main()
