"""
sim.py — a minimal simulator for the velocity-source Furuta plant.

Rolls plant.dynamics forward in time so controllers can be designed and validated
before touching hardware. The control input is the arm angular acceleration `a`
(= phi_ddot), held constant over each control step (zero-order hold), matching how
the discrete controller runs on the ESP32.

Optional saturation models the real motor: at 12 V the arm tops out at some max
speed, so `phi_dot_max` clips the arm velocity. Set it to the measured V_MAX once
calibration is done; leave None for unconstrained design.
"""
from __future__ import annotations

import numpy as np

import plant


class FurutaSim:
    def __init__(
        self,
        dt: float = plant.DT,
        substeps: int = 10,
        phi_dot_max: float | None = None,   # arm speed saturation [rad/s] (motor/12V limit)
    ):
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.phi_dot_max = phi_dot_max
        self.x = np.zeros(plant.N_STATE)
        self.t = 0.0

    def reset(self, theta0: float = 0.0, theta_dot0: float = 0.0,
              phi0: float = 0.0, phi_dot0: float = 0.0) -> np.ndarray:
        self.x = np.array([phi0, theta0, phi_dot0, theta_dot0], dtype=float)
        self.t = 0.0
        return self.x.copy()

    def step(self, a: float) -> np.ndarray:
        """Advance one control step under constant arm acceleration `a` (RK4)."""
        h = self.dt / self.substeps
        x = self.x
        for _ in range(self.substeps):
            k1 = plant.dynamics(x, a)
            k2 = plant.dynamics(x + 0.5 * h * k1, a)
            k3 = plant.dynamics(x + 0.5 * h * k2, a)
            k4 = plant.dynamics(x + h * k3, a)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if self.phi_dot_max is not None:
                x[plant.PHI_DOT] = np.clip(x[plant.PHI_DOT], -self.phi_dot_max, self.phi_dot_max)
        self.x = x
        self.t += self.dt
        return self.x.copy()

    @property
    def theta(self) -> float:
        return float(self.x[plant.THETA])


class ArmActuator:
    """Models the velocity-source Nidec: the controller asks for an arm acceleration
    a_des, but the firmware can only realize a QUANTIZED arm velocity (measured):
      |v| < v_deadzone        -> 0    (held; below this it stalls)
      v_deadzone <= |v| < v_min -> +-v_min  (snaps up to the min sustained ~3 rad/s)
      v_min <= |v| <= v_max   -> v     (tracks)
      |v| > v_max             -> +-v_max
    Given a_des and the current arm velocity, it returns the realized arm
    acceleration to feed the plant (so the sim sees what the motor really does).
    """

    def __init__(self, dt: float = plant.DT, v_deadzone: float = 1.5,
                 v_min: float = 2.98, v_max: float = 78.0, ideal: bool = False):
        self.dt = float(dt)
        self.v_deadzone = float(v_deadzone)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.ideal = bool(ideal)   # ideal=True -> perfect velocity source (no quantization)

    def _quantize(self, v: float) -> float:
        s = abs(v)
        if s < self.v_deadzone:
            return 0.0
        sg = 1.0 if v > 0 else -1.0
        if s < self.v_min:
            return sg * self.v_min
        if s > self.v_max:
            return sg * self.v_max
        return v

    def realize(self, v_cmd: float, phi_dot: float) -> float:
        """Given a commanded arm velocity, return the realized arm acceleration
        to feed the plant (quantized to the motor's real capability)."""
        v_real = v_cmd if self.ideal else self._quantize(v_cmd)
        v_real = float(np.clip(v_real, -self.v_max, self.v_max))
        return (v_real - phi_dot) / self.dt


if __name__ == "__main__":
    # Sanity: with no control, a tiny tilt should fall away from upright (diverge),
    # and the bottom (theta = pi) should be a stable rest it oscillates about.
    sim = FurutaSim()
    sim.reset(theta0=np.deg2rad(1.0))
    for _ in range(50):
        sim.step(0.0)
    print(f"uncontrolled from 1 deg after 50 steps: theta = {np.degrees(sim.theta):+.1f} deg",
          "(should grow -> upright unstable)")

    sim.reset(theta0=np.pi - np.deg2rad(5.0))   # near hanging (theta = pi is down)
    thetas = []
    for _ in range(600):
        sim.step(0.0)
        thetas.append(sim.theta)
    swing = (max(thetas) - min(thetas))
    print(f"uncontrolled near hanging: bounded oscillation, peak-to-peak "
          f"{np.degrees(swing):.1f} deg (should stay near the bottom, not diverge)")
