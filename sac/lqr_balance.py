"""
LQR balance controller for the Furuta pendulum.

A drop-in replacement for the SAC balancer: it outputs the same scalar command
u in [-1, 1] that run_swingup_balance.py already sends to the motor, so it slots
into the existing energy-swing-up -> handoff -> balance architecture unchanged.

How the gain is found (no training, no replay buffer, no collapse):
  1. Numerically linearize the *sim plant* about the upright equilibrium
     x* = [phi, theta, phi_dot, theta_dot] = 0, u* = 0, by central finite
     differences of one 100 Hz control step. This inherits the exact MuJoCo
     parameters and the voltage/torque actuator model used in training, so the
     gain matches the same physics the SAC policy was optimized against.
  2. Solve the discrete-time algebraic Riccati equation for the optimal gain K.
  3. Control law: u = -K (x - x_ref), clipped to the action limit.

The deadband / minimum-speed offset of the real drive is disabled *only* for the
linearization (it is a static input nonlinearity, not part of the dynamics). At
run time the command is the raw u, exactly as the SAC output was, so any downstream
deadband handling is identical to before.

State order: x = [phi, theta, phi_dot, theta_dot]  (theta = 0 is upright).
"""
from __future__ import annotations

import numpy as np
import mujoco
import scipy.linalg

from sim_balance_env import FurutaBalanceSimEnv


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _plant_step(env: FurutaBalanceSimEnv, x: np.ndarray, u: float) -> np.ndarray:
    """One control step of the (smooth) plant: x_{k+1} = f(x_k, u_k)."""
    env.data.qpos[0] = x[0]
    env.data.qpos[1] = x[1]
    env.data.qvel[0] = x[2]
    env.data.qvel[1] = x[3]
    env.data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env._prev_theta = float(x[1])
    env._theta_dot_filt = float(x[3])
    env._last_u = 0.0
    env.step(np.array([u], dtype=np.float32))
    return np.array(
        [float(env.data.qpos[0]), _wrap(float(env.data.qpos[1])),
         float(env.data.qvel[0]), float(env.data.qvel[1])]
    )


def linearize(env: FurutaBalanceSimEnv):
    """Central-difference linearization about upright -> discrete (A, B)."""
    # Disable the static input nonlinearity so the linearization is clean.
    env._deadband = 0.0
    env._min_frac = 0.0

    x0 = np.zeros(4)
    eps_x = np.array([1e-4, 1e-4, 1e-3, 1e-3])
    A = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4); dx[i] = eps_x[i]
        A[:, i] = (_plant_step(env, x0 + dx, 0.0) - _plant_step(env, x0 - dx, 0.0)) / (2 * eps_x[i])
    eps_u = 1e-3
    B = ((_plant_step(env, x0, eps_u) - _plant_step(env, x0, -eps_u)) / (2 * eps_u)).reshape(4, 1)
    return A, B


def design_lqr(Q: np.ndarray, R: np.ndarray, env: FurutaBalanceSimEnv | None = None):
    """Return (K, A, B) for the discrete LQR about upright."""
    own = env is None
    if own:
        env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=False)
    A, B = linearize(env)
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    if own:
        env.close()
    return K, A, B


def design_observer(A, B, C, Qn, Rn):
    """Steady-state Kalman predictor gain L for x_hat += L (y - C x_hat).

    Estimates the full state (incl. a clean theta_dot) from the clean position
    measurements, instead of trusting the laggy filtered velocity. This is the
    fix that makes LQR viable on this plant.
    """
    P = scipy.linalg.solve_discrete_are(A.T, C.T, Qn, Rn)
    L = (A @ P @ C.T) @ np.linalg.inv(C @ P @ C.T + Rn)
    return L


class LQRBalance:
    """u = -K (x - x_ref), x = [phi, theta, phi_dot, theta_dot]. Drop-in for SAC.

    The LQR is designed on the smooth plant (drive == action). At run time the
    desired drive is mapped back through the inverse of the motor's deadband /
    minimum-speed law so the realized drive matches the LQR intent, and a small
    deadzone suppresses corrections too fine for the >=min_frac minimum kick
    (otherwise tiny tilts trigger an over-strong kick and the loop oscillates).
    """

    # Weights: theta dominates; strong theta_dot damping to tolerate the filtered
    # (laggy) velocity; arm lightly regulated to stay centered. Higher R keeps the
    # commands gentle so the coarse actuator does not overshoot.
    DEFAULT_Q = np.diag([0.5, 60.0, 0.2, 8.0])
    DEFAULT_R = np.array([[6.0]])

    # Observer noise: trust the clean positions (small Rn), let the velocities
    # carry the process uncertainty (larger Qn on the velocity states).
    DEFAULT_QN = np.diag([1e-3, 1e-3, 1e-1, 1e-1])
    DEFAULT_RN = np.diag([1e-5, 1e-5, 1e-3])

    def __init__(self, Q=None, R=None, action_limit: float = 1.0,
                 min_frac: float = 0.15, deadband: float = 0.05, deadzone: float = 0.03,
                 compensate: bool = True, observer: bool = True,
                 env: FurutaBalanceSimEnv | None = None):
        self.Q = np.asarray(self.DEFAULT_Q if Q is None else Q, dtype=float)
        self.R = np.asarray(self.DEFAULT_R if R is None else R, dtype=float)
        self.action_limit = float(action_limit)
        self._min_frac = float(min_frac)
        self._deadband = float(deadband)
        self._deadzone = float(deadzone)
        self._compensate = bool(compensate)
        self._use_observer = bool(observer)
        own = env is None
        if own:
            env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=False)
        self.K, self.A, self.B = design_lqr(self.Q, self.R, env=env)
        # Measure clean positions + arm velocity; estimate theta_dot.
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        self.L = design_observer(self.A, self.B, self.C, self.DEFAULT_QN, self.DEFAULT_RN)
        if own:
            env.close()
        self._xhat = None

    def reset(self):
        self._xhat = None

    def _to_action(self, u_des: float) -> float:
        """Map a desired drive to the motor command that realizes it."""
        d = abs(u_des)
        if d < self._deadzone:
            return 0.0
        if not self._compensate:
            return float(np.clip(u_des, -self.action_limit, self.action_limit))
        a = (d - self._min_frac) / (1.0 - self._min_frac)   # invert min-speed law
        a = max(a, self._deadband + 1e-3)                   # clear the firmware deadband
        return float(np.sign(u_des) * min(a, self.action_limit))

    def _realized_drive(self, a: float) -> float:
        """Actual drive the motor produces for command a (matches the sim plant)."""
        if abs(a) < self._deadband:
            return 0.0
        return float(np.sign(a) * (self._min_frac + abs(a) * (1.0 - self._min_frac)))

    def __call__(self, obs: np.ndarray) -> float:
        cos_th, sin_th, th_dot, phi, phi_dot = (float(v) for v in obs[:5])
        theta = float(np.arctan2(sin_th, cos_th))   # 0 = upright
        if not self._use_observer:
            x = np.array([phi, theta, phi_dot, th_dot])
            return self._to_action(float(-(self.K @ x).item()))

        y = np.array([phi, theta, phi_dot])          # clean measurements
        if self._xhat is None:                        # init from first obs
            self._xhat = np.array([phi, theta, phi_dot, th_dot], dtype=float)
        u_des = float(-(self.K @ self._xhat).item())
        action = self._to_action(u_des)
        drive = self._realized_drive(action)   # actual drive applied, not the intent
        # Predict + correct on the drive that was really commanded.
        self._xhat = self.A @ self._xhat + self.B.flatten() * drive + self.L @ (y - self.C @ self._xhat)
        return action


# ------------------------------------------------------------------
def _hold_duration(env, ctrl, theta0, theta_dot0, phi0=0.0) -> float:
    obs, _ = env.reset(seed=0, options={"theta": theta0, "theta_dot": theta_dot0, "phi": phi0})
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    streak, best, done = 0, 0, False
    while not done:
        obs, _, terminated, truncated, _ = env.step(np.array([ctrl(obs)], dtype=np.float32))
        th = float(np.arctan2(obs[1], obs[0]))
        if abs(th) < np.deg2rad(10.0) and abs(obs[2]) < 3.0:
            streak += 1; best = max(best, streak)
        else:
            streak = 0
        done = terminated or truncated
    return best * env.dt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Design + sim-validate the LQR balance controller.")
    ap.add_argument("--angle-max", type=float, default=20.0)
    ap.add_argument("--vel-max", type=float, default=8.0)
    ap.add_argument("--n-angle", type=int, default=13)
    ap.add_argument("--n-vel", type=int, default=13)
    ap.add_argument("--hold-seconds", type=float, default=2.0)
    args = ap.parse_args()

    ctrl = LQRBalance()
    A, B, K = ctrl.A, ctrl.B, ctrl.K
    cl_eig = np.linalg.eigvals(A - B @ K)
    print("Discrete A eigenvalues (open loop):", np.round(np.linalg.eigvals(A), 3))
    print("  |.| :", np.round(np.abs(np.linalg.eigvals(A)), 3), " (any > 1 => unstable upright)")
    print("Closed-loop eigenvalues |.|     :", np.round(np.abs(cl_eig), 3),
          " (all < 1 => stabilized)")
    print("Gain K [phi, theta, phi_dot, theta_dot] =", np.round(K.flatten(), 3), "\n")

    # Region-of-attraction sweep, torque model, no DR (compare to roa_sweep.py).
    env = FurutaBalanceSimEnv(domain_rand=False, velocity_control=False,
                              episode_seconds=8.0, fall_threshold_deg=55.0)
    angles = np.linspace(-args.angle_max, args.angle_max, args.n_angle)
    vels = np.linspace(args.vel_max, -args.vel_max, args.n_vel)
    grid = np.zeros((args.n_vel, args.n_angle))
    header = "thd\\th |" + "".join(f"{a:6.0f}" for a in angles)
    print(header); print("-" * len(header))
    for i, thd in enumerate(vels):
        row = ""
        for j, a in enumerate(angles):
            h = _hold_duration(env, ctrl, np.deg2rad(a), float(thd))
            grid[i, j] = h
            row += f"{h:4.1f}{'#' if h >= args.hold_seconds else ' '}"
        print(f"{thd:+6.1f} |" + row)
    env.close()
    ok = int((grid >= args.hold_seconds).sum())
    print(f"\nGrid success (>= {args.hold_seconds:.1f}s): {ok}/{grid.size} cells "
          f"({100.0*ok/grid.size:.0f}%)   median hold: {np.median(grid):.2f}s")
