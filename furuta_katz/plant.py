"""
plant.py — the physics model of the Furuta pendulum (velocity-source motor).

Single source of truth for the dynamics. No hardware, no controller, no time
stepping: it only answers "given the state and the arm acceleration, how is the
system accelerating right now?". sim.py integrates this; balance.py linearizes it.

Why this model (and not Ben Katz's torque model):
  The Nidec 24H404H160 has an integrated speed controller — we command arm SPEED,
  not torque. Its internal loop forces the arm to the commanded velocity, so the
  arm is a *commanded input*, not a free body. That collapses the full 2-DOF
  coupled Furuta dynamics into a single pendulum-on-an-accelerating-base equation:

      theta_ddot = alpha*sin(theta) - beta*cos(theta) * a

  where a = phi_ddot is the commanded arm angular acceleration and theta is the
  pendulum angle measured from UPRIGHT (theta = 0 is up). Approximation: assumes
  the speed loop is much faster than the pendulum and drops the small centrifugal
  (phi_dot^2) coupling — both fine near upright for this short, fast rod.

State (used everywhere):
    x = [phi, theta, phi_dot, theta_dot]
        phi       arm angle            [rad]
        theta     pendulum from upright [rad]   (0 = up, +-pi = hanging)
        phi_dot   arm rate             [rad/s]
        theta_dot pendulum rate        [rad/s]
Input:
    a = phi_ddot  commanded arm angular acceleration [rad/s^2]
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

# --- geometry (uniform rod, no tip mass) ---------------------------------
G = 9.81          # gravity                       [m/s^2]
L_ROD = 0.075     # pendulum rod length           [m]
L_ARM = 0.035     # arm pivot radius              [m]
# mass cancels for a uniform rod, so it never appears below.

# --- model coefficients (pure geometry) ----------------------------------
# alpha = m*g*l_cm / J = 3g/(2L)      gravity / instability term  [1/s^2]
# beta  = m*l_cm*L_arm / J = 3*L_arm/(2L)   control authority     [dimensionless]
# BETA sign measured FLIPPED on hardware (sign_check.py: +arm velocity -> -theta_dot),
# so it is negative; magnitude is pure geometry. This makes the LQR and swing-up steer
# the correct way on the real rig.
ALPHA = 3.0 * G / (2.0 * L_ROD)          # ~= 196.2
BETA = -3.0 * L_ARM / (2.0 * L_ROD)      # ~= -0.70  (hardware coupling sign)

# --- control timing (measured ESP32 rate, see check_rate.py) --------------
CONTROL_HZ = 166.0
DT = 1.0 / CONTROL_HZ                     # ~= 0.006 s

# state indices, for readability
PHI, THETA, PHI_DOT, THETA_DOT = 0, 1, 2, 3
N_STATE = 4


def dynamics(x: np.ndarray, a: float) -> np.ndarray:
    """Continuous-time state derivative xdot = f(x, a). Instantaneous, no dt."""
    theta = x[THETA]
    phi_ddot = a                                                  # velocity-source arm
    theta_ddot = ALPHA * np.sin(theta) - BETA * np.cos(theta) * a
    return np.array([x[PHI_DOT], x[THETA_DOT], phi_ddot, theta_ddot], dtype=float)


def pendulum_energy(theta: float, theta_dot: float) -> float:
    """Pendulum mechanical energy per unit (m*l_cm), referenced so upright is max.

    E = 1/2 * (J / (m*l_cm)) * theta_dot^2 + g*cos(theta), but since alpha =
    m*g*l_cm/J we use the equivalent scaled form that swing-up needs:
        E_scaled = 0.5 * theta_dot^2 / alpha + cos(theta)        [up = +1]
    Returned in these scaled units so the swing-up gain is geometry-independent.
    """
    return 0.5 * theta_dot ** 2 / ALPHA + np.cos(theta)


# energy of the upright rest state (theta=0, theta_dot=0), in the same units
E_UPRIGHT = 1.0


def linearize() -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time linearization about upright (x*=0, a*=0): xdot = A x + B a.

    sin(theta)~theta, cos(theta)~1  =>  theta_ddot ~= alpha*theta - beta*a.
    """
    A = np.array([
        [0.0, 0.0,   1.0, 0.0],
        [0.0, 0.0,   0.0, 1.0],
        [0.0, 0.0,   0.0, 0.0],
        [0.0, ALPHA, 0.0, 0.0],
    ])
    B = np.array([[0.0], [0.0], [1.0], [-BETA]])
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization via the augmented-matrix exponential.

    expm([[A, B], [0, 0]] * dt) = [[Ad, Bd], [0, I]].  Works even though A is
    singular (the double integrator on phi has zero eigenvalues).
    """
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = scipy.linalg.expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    A, B = linearize()
    Ad, Bd = discretize(A, B)

    print("Model coefficients")
    print(f"  ALPHA = {ALPHA:.3f} 1/s^2   (sqrt = {np.sqrt(ALPHA):.2f} rad/s natural rate)")
    print(f"  BETA  = {BETA:.3f}")
    print(f"  DT    = {DT:.5f} s  ({CONTROL_HZ:.0f} Hz)\n")

    print("Continuous A:\n", A)
    print("B:\n", B.ravel())
    eig_c = np.linalg.eigvals(A)
    print("\nContinuous eigenvalues:", np.round(eig_c, 3))
    print("  -> a positive real eigenvalue means UPRIGHT IS UNSTABLE (expected).")

    eig_d = np.linalg.eigvals(Ad)
    print("\nDiscrete |eigenvalues|:", np.round(np.abs(eig_d), 4))
    print("  -> any > 1 confirms the open-loop instability the LQR must fix.")

    # controllability: can the input reach every state? (rank of [B AB A^2B A^3B])
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(N_STATE)])
    print(f"\nControllability matrix rank: {np.linalg.matrix_rank(C)} / {N_STATE}",
          "(must be 4 for LQR to work)")
