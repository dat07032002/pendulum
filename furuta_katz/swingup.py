"""
swingup.py — energy-shaping swing-up for the velocity-source Furuta plant.

Ben Katz's energy pump, adapted to a velocity-source motor. We add energy to the
pendulum until it reaches the upright energy, then hand off to the LQR balance.

Derivation (scaled energy E = 0.5*theta_dot^2/alpha + cos(theta), upright E = +1):
    dE/dt = -(beta/alpha) * a * theta_dot * cos(theta)
so to drive E up toward E_upright we choose the arm acceleration
    a = -k_e * (E_upright - E) * theta_dot * cos(theta)
which makes dE/dt = (beta/alpha)*k_e*(E_upright-E)*(theta_dot*cos theta)^2 >= 0
whenever the pendulum is below upright energy. Output is an arm acceleration
`a` (rad/s^2), same units as balance.py, so hw_env integrates it the same way.

Cable-wrap constraint (your rig, not Ben's slip ring): the arm can only travel a
limited range, so the law also softly centers the arm and reflects it back before
it hits the limit. Sign of the energy term assumes the model's beta>0; verify on
hardware with a sign check before trusting it.
"""
from __future__ import annotations

import numpy as np

import plant


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class EnergySwingUp:
    """Energy-regulated proportional swing-up (Astrom). Returns an arm VELOCITY [rad/s].

    Pump law:  v = -sign(beta) * k_e * theta_dot * cos(theta) * dE,   dE = E_up - E.
      dE > 0 (below upright energy)  -> pumps energy in.
      dE < 0 (OVER-energy)           -> v flips sign -> BRAKES energy out.
    Coasting only in a TIGHT band |dE| < coast_band (energy matched), so the rod is
    regulated to exactly the upright energy and arrives at the top SLOWLY (homoclinic
    orbit) instead of flying over -- which is what makes it catchable. The proportional
    (vs bang-bang) form also tapers near the top for a gentle final approach.
    sign(plant.BETA) sets the coupling direction; needs the fast arm (firmware DUTY_MAX).
    """

    def __init__(
        self,
        k_e: float = 6.0,             # energy-pump gain  [ (rad/s) per (energy*rad/s) ]
        v_max: float = 45.0,          # arm-speed ceiling during swing-up [rad/s] (fast!)
        coast_band: float = 0.06,     # coast (let catch take over) when |dE| < this
        k_center: float = 2.0,        # gentle pull toward arm center (bounds drift to +-180)
    ):
        self.k_e = float(k_e)
        self.v_max = float(v_max)
        self.coast_band = float(coast_band)
        self.k_center = float(k_center)

    def reset(self):
        pass

    def __call__(self, x: np.ndarray) -> float:
        phi = float(x[plant.PHI])
        theta = _wrap(float(x[plant.THETA]))
        theta_dot = float(x[plant.THETA_DOT])

        E = plant.pendulum_energy(theta, theta_dot)
        dE = plant.E_UPRIGHT - E                  # deficit (>0 below, <0 over-energy)
        if abs(dE) < self.coast_band:
            v = 0.0                                # energy matched: coast, let the catch take over
        else:
            # pumps when dE>0, BRAKES when dE<0 (the key to a slow arrival)
            v = -np.sign(plant.BETA) * self.k_e * theta_dot * np.cos(theta) * dE
        v -= self.k_center * phi                   # gentle centering: keep the arm within +-180 deg
        return float(np.clip(v, -self.v_max, self.v_max))


# ------------------------------------------------------------------
if __name__ == "__main__":
    from sim import FurutaSim, ArmActuator

    sim = FurutaSim(phi_dot_max=150.0)
    su = EnergySwingUp()
    act = ArmActuator()                    # real actuator (deadzone + 3 rad/s floor)

    # Start hanging (theta = pi = down), small nudge to break symmetry.
    x = sim.reset(theta0=np.pi - np.deg2rad(1.0))   # near rest, like hardware
    handoff = np.deg2rad(15.0)
    reached_at = None
    max_phi = 0.0
    A, ret, vcap = np.deg2rad(80.0), 8.0, 45.0       # match run.py clamp
    booted = False
    for k in range(int(10.0 / sim.dt)):    # up to 10 s
        amp_b = np.pi - abs(_wrap(float(x[plant.THETA])))
        if amp_b >= np.deg2rad(30.0):
            booted = True
        phi = float(x[plant.PHI])
        if not booted:
            tgt = np.deg2rad(45.0) * np.sign(np.sin(2 * np.pi * (k * sim.dt) / (2 * np.pi / np.sqrt(plant.ALPHA))))
            v_cmd = 20.0 * (tgt - phi)
        else:
            v_cmd = su(x)
        v_cmd = float(np.clip(v_cmd, -vcap, vcap))
        if phi > A:
            v_cmd = -ret
        elif phi < -A:
            v_cmd = +ret
        a = act.realize(v_cmd, float(x[plant.PHI_DOT]))
        x = sim.step(a)
        max_phi = max(max_phi, abs(x[plant.PHI]))
        if abs(_wrap(x[plant.THETA])) < handoff:
            reached_at = k * sim.dt
            break

    th = np.degrees(_wrap(x[plant.THETA]))
    thd = x[plant.THETA_DOT]
    print(f"k_e={su.k_e}, v_max={su.v_max}, coast_band={su.coast_band}")
    if reached_at is not None:
        print(f"reached handoff band (15 deg) at t = {reached_at:.2f} s")
        print(f"  state at handoff: theta = {th:+.1f} deg, theta_dot = {thd:+.2f} rad/s")
    else:
        print(f"did NOT reach handoff in 8 s; ended at theta = {th:+.1f} deg")
    print(f"  max arm excursion during swing-up: {np.degrees(max_phi):.0f} deg "
          f"(cable limit ~120 deg)")
    print(f"  handoff theta_dot must be catchable by balance (it tolerates ~|2| rad/s near upright)")
