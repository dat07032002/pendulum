"""
balance.py — LQR balance controller for the velocity-source Furuta plant.

Ben Katz's "linear controller near upright", adapted to a velocity-source motor:
the LQR is designed on plant.linearize() with the arm acceleration `a` as the
input, so the output is an acceleration (rad/s^2), not a torque. On hardware this
`a` is integrated to an arm speed setpoint and mapped to the motor command `u`
(done later in hw_env.py). Here, in sim, the plant integrates `a` itself.

Control law:
    a = -K (x - x_ref),   x = [phi, theta, phi_dot, theta_dot],  x_ref = 0
clipped to a_max so a single step cannot demand an impossible kick.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

import plant


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def design_lqr(Q: np.ndarray, R: np.ndarray, dt: float = plant.DT):
    """Discrete LQR about upright. Returns (K, Ad, Bd)."""
    A, B = plant.linearize()
    Ad, Bd = plant.discretize(A, B, dt)
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, Ad, Bd


def design_observer(Ad, C, Qn, Rn):
    """Steady-state Kalman predictor gain L for x_hat += L (y - C x_hat).
    Estimates a CLEAN theta_dot from the clean position measurements, instead of
    trusting the laggy/quantized firmware derivative -- the fix that makes the LQR
    stable on real hardware."""
    P = scipy.linalg.solve_discrete_are(Ad.T, C.T, Qn, Rn)
    L = (Ad @ P @ C.T) @ np.linalg.inv(C @ P @ C.T + Rn)
    return L


class LQRBalance:
    """a = -K x. Drop-in: __call__ takes the 4-state x, returns arm accel `a`."""

    # Weights, tuned in sim against the ~120 deg cable-wrap limit. Unlike Ben's
    # slip-ring rig (no arm-position term at all), we must STRONGLY center the arm
    # (large q_phi) or it winds past the cable limit while balancing. Low R makes
    # the catch aggressive/tight, which keeps the arm EXCURSION small (a slow,
    # gentle catch sweeps the arm much further). Validated: <120 deg excursion
    # catching up to a 20 deg tilt.
    DEFAULT_Q = np.diag([400.0, 120.0, 10.0, 12.0])   # [phi, theta, phi_dot, theta_dot]
    DEFAULT_R = np.array([[0.5]])
    # Observer noise: trust the clean positions (small Rn), let the velocities carry
    # the process uncertainty. C measures [phi, theta, phi_dot]; theta_dot is estimated.
    DEFAULT_QN = np.diag([1e-3, 1e-3, 1e-1, 2e-1])
    DEFAULT_RN = np.diag([1e-5, 1e-5, 5e-3])

    def __init__(self, Q=None, R=None, a_max: float = 400.0, dt: float = plant.DT,
                 observer: bool = True):
        self.Q = np.asarray(self.DEFAULT_Q if Q is None else Q, dtype=float)
        self.R = np.asarray(self.DEFAULT_R if R is None else R, dtype=float)
        self.a_max = float(a_max)
        self.dt = float(dt)
        self.K, self.Ad, self.Bd = design_lqr(self.Q, self.R, self.dt)
        self._use_obs = bool(observer)
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        self.L = design_observer(self.Ad, self.C, self.DEFAULT_QN, self.DEFAULT_RN)
        self._xhat = None

    def set_weights(self, q_phi=None, q_theta=None, q_phidot=None, q_thetadot=None, R=None):
        """Live-update LQR weights and rebuild the gain K."""
        qd = np.diag(self.Q).astype(float).copy()
        for i, v in enumerate((q_phi, q_theta, q_phidot, q_thetadot)):
            if v is not None:
                qd[i] = float(v)
        self.Q = np.diag(qd)
        if R is not None:
            self.R = np.array([[float(R)]])
        self.K, self.Ad, self.Bd = design_lqr(self.Q, self.R, self.dt)
        return self.K

    def reset(self):
        self._xhat = None

    def __call__(self, x: np.ndarray) -> float:
        xv = np.array([x[plant.PHI], _wrap(x[plant.THETA]),
                       x[plant.PHI_DOT], x[plant.THETA_DOT]], dtype=float)
        if not self._use_obs:
            a = float(-(self.K @ xv).item())
            return float(np.clip(a, -self.a_max, self.a_max))
        # Observer: estimate a clean state (esp. theta_dot) from the position measurements.
        y = xv[:3]                                  # measured [phi, theta, phi_dot]
        if self._xhat is None:
            self._xhat = xv.copy()
        a = float(-(self.K @ self._xhat).item())
        a = float(np.clip(a, -self.a_max, self.a_max))
        # predict with the commanded accel, correct from the clean position measurements
        self._xhat = self.Ad @ self._xhat + self.Bd.flatten() * a + self.L @ (y - self.C @ self._xhat)
        return a


# ------------------------------------------------------------------
def _catch(sim, ctrl, theta0, theta_dot0=0.0, seconds=3.0):
    """Run a catch from a tilt; return (held, stats). 'held' = ended upright+slow."""
    from sim import FurutaSim  # local import to avoid a cycle if reused
    x = sim.reset(theta0=theta0, theta_dot0=theta_dot0)
    ctrl.reset()
    n = int(seconds / sim.dt)
    max_a = max_phidot = max_phi = 0.0
    held_streak = best_streak = 0
    for _ in range(n):
        a = ctrl(x)
        x = sim.step(a)
        max_a = max(max_a, abs(a))
        max_phidot = max(max_phidot, abs(x[plant.PHI_DOT]))
        max_phi = max(max_phi, abs(x[plant.PHI]))
        if abs(_wrap(x[plant.THETA])) < np.deg2rad(8.0) and abs(x[plant.THETA_DOT]) < 2.0:
            held_streak += 1
            best_streak = max(best_streak, held_streak)
        else:
            held_streak = 0
    held = abs(_wrap(x[plant.THETA])) < np.deg2rad(5.0) and abs(x[plant.THETA_DOT]) < 1.0
    return held, dict(max_a=max_a, max_phidot=max_phidot, max_phi=max_phi,
                      best_hold=best_streak * sim.dt)


if __name__ == "__main__":
    from sim import FurutaSim

    np.set_printoptions(precision=4, suppress=True)
    ctrl = LQRBalance()
    K = ctrl.K
    cl = ctrl.Ad - ctrl.Bd @ K
    eig = np.linalg.eigvals(cl)

    print("LQR gain K [phi, theta, phi_dot, theta_dot] =", np.round(K.ravel(), 3))
    print("Closed-loop discrete |eigenvalues| =", np.round(np.abs(eig), 4))
    print("  -> all < 1 means the LQR stabilizes upright.\n")

    # 12 V motor: cap the arm speed so the sim is honest about reachability.
    # Placeholder until calibration; ~150 rad/s is generous for a 12 V Nidec arm.
    sim = FurutaSim(phi_dot_max=150.0)

    print("Catch test (start tilted, theta_dot = 0):")
    print(f"{'tilt deg':>8} | {'held?':>5} | {'hold s':>6} | {'max a':>7} | "
          f"{'max phidot':>10} | {'max phi deg':>11}")
    print("-" * 64)
    for deg in (5, 10, 15, 20, 25, 30, 40):
        held, s = _catch(sim, ctrl, np.deg2rad(deg))
        print(f"{deg:8d} | {('YES' if held else 'no'):>5} | {s['best_hold']:6.2f} | "
              f"{s['max_a']:7.1f} | {s['max_phidot']:10.2f} | "
              f"{np.degrees(s['max_phi']):11.1f}")

    print("\nNote: max phi (arm excursion) must stay well under the ~120 deg cable-wrap")
    print("limit; max phidot must be reachable by the 12 V motor (check vs V_MAX later).")
