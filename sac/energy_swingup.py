"""
Åström energy-based swing-up controller for the Furuta pendulum.

The pendulum's mechanical energy is compared to the target energy at the
upright equilibrium.  The arm is kicked in the direction that injects energy
when the pendulum is in the right phase.

Control law:
    u = clip(k_e * theta_dot * cos(theta) * dE, -u_max, u_max)

where  dE = E - E_ref >= 0  (zero only at the upright equilibrium).

Physical parameters match the hardware: rod 75 mm, 25 g.
"""
from __future__ import annotations
import numpy as np


class EnergySwingUp:
    # Hardware-measured pendulum parameters
    M_ROD = 0.025           # rod mass            [kg]
    L_ROD = 0.075           # rod length          [m]
    L_CM  = L_ROD / 2       # CoM from elbow      [m]
    I_ROD = M_ROD * L_ROD**2 / 3   # inertia about elbow  [kg m^2]
    G     = 9.81            # gravity             [m/s^2]

    # Total energy range: hanging (dE = E_max) -> upright (dE = 0)
    @property
    def E_max(self) -> float:
        return 2.0 * self.M_ROD * self.G * self.L_CM   # ~0.0184 J for this hardware

    def __init__(
        self,
        k_e: float = 0.8,
        u_max: float = 0.8,
        phi_limit_deg: float = 80.0,
        coast_fraction: float = 0.15,
        k_center: float = 0.15,
        k_brake: float = 20.0,
        brake_umax: float = 0.8,
    ):
        """
        k_e             : energy-pump gain when dE > 0 (below target energy).
        u_max           : arm command ceiling during pumping  [0..1].
        phi_limit_deg   : arm travel limit; prevents cable wrap.
        coast_fraction  : legacy passive-coast threshold (ignored when k_brake>0
                          because the asymmetric law handles braking automatically).
        k_center        : arm-centering gain.
        k_brake         : braking gain when dE < 0 (pendulum has excess energy).
                          Asymmetric Åström: pump gently with k_e, brake hard with
                          k_brake.  Kicks in automatically the moment the pendulum
                          overshoots E_ref — no angle threshold needed.
                          Try 15–30.  0 = revert to old coast-zone behaviour.
        brake_umax      : arm command ceiling during braking [0..1].
        """
        self.k_e            = float(k_e)
        self.u_max          = float(u_max)
        self.phi_limit      = np.deg2rad(float(phi_limit_deg))
        self.coast_fraction = float(coast_fraction)
        self.k_center       = float(k_center)
        self.k_brake        = float(k_brake)
        self.brake_umax     = float(brake_umax)
        self.E_ref          = -self.M_ROD * self.G * self.L_CM

    def pendulum_energy(self, cos_th: float, th_dot: float) -> float:
        """Mechanical energy with theta=0 at upright."""
        return 0.5 * self.I_ROD * th_dot**2 - self.M_ROD * self.G * self.L_CM * cos_th

    def __call__(self, obs: np.ndarray) -> float:
        """
        obs : [cos_theta, sin_theta, theta_dot, phi, phi_dot]
        returns : arm command u in [-u_max, +u_max]
        """
        cos_th, _sin_th, th_dot, phi, _phi_dot = map(float, obs)

        E  = self.pendulum_energy(cos_th, th_dot)
        dE = E - self.E_ref   # >= 0; zero only at the upright equilibrium

        if self.k_brake > 0.0:
            # Asymmetric Åström: pump gently when below target, brake hard when above.
            # dE > 0: pendulum needs more energy -> pump with k_e.
            # dE < 0: pendulum has excess energy at upright -> brake with k_brake.
            # The sign of (th_dot * cos_th * dE) automatically points the right way.
            if dE >= 0:
                u = float(np.clip(self.k_e * th_dot * cos_th * dE, -self.u_max, self.u_max))
            else:
                u = float(np.clip(self.k_brake * th_dot * cos_th * dE,
                                  -self.brake_umax, self.brake_umax))
        else:
            # Legacy coast-zone mode (k_brake=0).
            if dE < self.coast_fraction * self.E_max:
                u = 0.0
            else:
                u = float(np.clip(self.k_e * th_dot * cos_th * dE, -self.u_max, self.u_max))

        # Arm centering: gentle restoring force toward phi=0 to counteract
        # cable spring bias that causes continuous rotation.
        u -= self.k_center * phi

        u = float(np.clip(u, -self.u_max, self.u_max))

        # Hard arm travel limit: strong reversal if past the boundary.
        if abs(phi) > self.phi_limit and u * phi > 0.0:
            u = -0.5 * self.u_max * float(np.sign(phi))

        return u
