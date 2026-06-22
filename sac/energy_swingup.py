"""
Energy-based swing-up controller for the Furuta pendulum.

The pendulum's mechanical energy is compared to the target energy at the
upright equilibrium. The arm is kicked in the direction that injects energy
when the pendulum is in the right phase.

Control law:
    u = clip(k_e * theta_dot * cos(theta) * dE, -u_max, u_max)

where dE is the energy deficit. The controller pumps while the pendulum
needs energy, then coasts near/above the target energy.

Physical parameters match the hardware: rod 75 mm, 25 g.
"""
from __future__ import annotations

import numpy as np


class EnergySwingUp:
    # Hardware-measured pendulum parameters
    M_ROD = 0.025
    L_ROD = 0.075
    L_CM = L_ROD / 2
    I_ROD = M_ROD * L_ROD**2 / 3
    G = 9.81

    @property
    def E_max(self) -> float:
        """Energy gap between hanging and upright rest."""
        return 2.0 * self.M_ROD * self.G * self.L_CM

    def __init__(
        self,
        k_e: float = 0.8,
        u_max: float = 0.8,
        phi_limit_deg: float = 80.0,
        coast_fraction: float = 0.15,
        k_center: float = 0.15,
        k_arm_damp: float = 0.0,
        u_floor: float = 0.0,
    ):
        """
        k_e            : energy-pump gain when dE > 0.
        u_max          : arm command ceiling during swing-up [0..1].
        phi_limit_deg  : arm travel limit; prevents cable wrap.
        coast_fraction : coast when energy deficit falls below this fraction
                         of the hanging-to-upright energy range.
        k_center       : arm-centering gain that subtracts k_center*phi.
        k_arm_damp     : arm velocity damping gain that subtracts
                         k_arm_damp*phi_dot.
        u_floor        : optional minimum non-zero command after the energy
                         law picks a direction.
        """
        self.k_e = float(k_e)
        self.u_max = float(u_max)
        self.phi_limit = np.deg2rad(float(phi_limit_deg))
        self.coast_fraction = float(coast_fraction)
        self.k_center = float(k_center)
        self.k_arm_damp = float(k_arm_damp)
        self.u_floor = float(np.clip(abs(u_floor), 0.0, self.u_max))
        self.E_ref = self.M_ROD * self.G * self.L_CM

    def pendulum_energy(self, cos_th: float, th_dot: float) -> float:
        """Mechanical energy, potential maximum at upright (cos theta = +1)."""
        return 0.5 * self.I_ROD * th_dot**2 + self.M_ROD * self.G * self.L_CM * cos_th

    def _apply_floor(self, u: float) -> float:
        if self.u_floor <= 0.0 or u == 0.0:
            return u
        return float(np.sign(u) * max(abs(u), self.u_floor))

    def __call__(self, obs: np.ndarray) -> float:
        """
        obs : [cos_theta, sin_theta, theta_dot, phi, phi_dot]
        returns : arm command u in [-u_max, +u_max]
        """
        # Use the first five fields only; the SAC obs may append extra channels
        # (e.g. previous action) that the energy law does not consume.
        cos_th, _sin_th, th_dot, phi, phi_dot = map(float, obs[:5])

        energy = self.pendulum_energy(cos_th, th_dot)
        dE = self.E_ref - energy

        if dE < self.coast_fraction * self.E_max:
            u = 0.0
        else:
            u = float(np.clip(self.k_e * th_dot * cos_th * dE, -self.u_max, self.u_max))

        u = self._apply_floor(u)

        # Keep the arm from winding while avoiding active pendulum braking.
        u -= self.k_center * phi
        u -= self.k_arm_damp * phi_dot
        u = float(np.clip(u, -self.u_max, self.u_max))

        if abs(phi) > self.phi_limit and u * phi > 0.0:
            u = -0.5 * self.u_max * float(np.sign(phi))

        return u
