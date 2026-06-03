"""
Furuta Pendulum Simulation — LQR Control
Parameters extracted from SolidWorks CAD model
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ══════════════════════════════════════════════
# SYSTEM PARAMETERS (from SolidWorks)
# ══════════════════════════════════════════════
m1   = 0.03478    # arm mass [kg]
m2   = 0.02646    # pendulum mass [kg]
I1   = 1.431e-5   # arm MOI about motor shaft [kg·m²]
I2   = 6.721e-5   # pendulum MOI about pivot [kg·m²]
L1   = 0.04239    # arm length: motor shaft → pendulum pivot [m]
L2   = 0.06772    # pendulum length: pivot → tip [m]
Lc2  = 0.04714    # pendulum COM distance from pivot [m]
g    = 9.81       # gravity [m/s²]
Rm   = 8.4        # motor resistance [Ω]  (Pololu HPCB typical)
kt   = 0.017      # motor torque constant [N·m/A]
km   = 0.017      # back-EMF constant [V·s/rad]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════
# LINEARIZATION AT UPRIGHT EQUILIBRIUM
# State: x = [theta, alpha, dtheta, dalpha]
# alpha=0 → pendulum upright
# ══════════════════════════════════════════════
def get_linear_system():
    M11 = I1 + m2 * L1**2
    M12 = m2 * L1 * Lc2
    M22 = I2 + m2 * Lc2**2

    M = np.array([[M11, M12],
                  [M12, M22]])
    Minv = np.linalg.inv(M)

    # Gravity term linearized about alpha=0 (upright)
    K_grav = np.array([[0,            0],
                       [0, -m2 * g * Lc2]])

    # Input: torque on arm axis only
    B_q = np.array([[1], [0]])

    A = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [Minv @ (-K_grav), np.zeros((2, 2))]])

    B = np.block([[np.zeros((2, 1))],
                  [Minv @ B_q]])

    return A, B

# ══════════════════════════════════════════════
# LQR DESIGN
# ══════════════════════════════════════════════
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

A, B = get_linear_system()

# Tune Q: penalize alpha (pendulum angle) most heavily
Q = np.diag([1.0,   # theta      — arm angle
             50.0,  # alpha      — pendulum angle (most important)
             1.0,   # dtheta     — arm velocity
             5.0])  # dalpha     — pendulum velocity
R = np.array([[0.5]])  # control effort

K = lqr(A, B, Q, R)
print("LQR gain K:", np.round(K, 4))
print("Eigenvalues (closed-loop):", np.round(
    np.linalg.eigvals(A - B @ K), 3))

# ══════════════════════════════════════════════
# NONLINEAR DYNAMICS
# ══════════════════════════════════════════════
def furuta_dynamics(t, x, u):
    """
    Full nonlinear EOM for Furuta pendulum.
    x = [theta, alpha, dtheta, dalpha]
    u = motor torque [N·m]
    alpha = 0 → upright (unstable equilibrium)
    """
    _, alpha, dtheta, dalpha = x

    # Mass matrix (state-dependent)
    M11 = I1 + m2 * L1**2 + m2 * Lc2**2 * np.sin(alpha)**2
    M12 = m2 * L1 * Lc2 * np.cos(alpha)
    M22 = I2 + m2 * Lc2**2
    M_mat = np.array([[M11, M12],
                      [M12, M22]])

    # Coriolis + centripetal
    C1 = (-m2 * Lc2**2 * np.sin(2*alpha) * dtheta * dalpha
          - 0.5 * m2 * Lc2**2 * np.sin(2*alpha) * dalpha**2)
    C2 = (0.5 * m2 * Lc2**2 * np.sin(2*alpha) * dtheta**2
          - m2 * L1 * Lc2 * np.sin(alpha) * dtheta**2)

    # Gravity
    G1 = 0.0
    G2 = -m2 * g * Lc2 * np.sin(alpha)

    # RHS
    tau = np.array([u, 0.0])
    C_vec = np.array([C1, C2])
    G_vec = np.array([G1, G2])

    rhs = tau - C_vec - G_vec
    qdd = np.linalg.solve(M_mat, rhs)

    return [dtheta, dalpha, qdd[0], qdd[1]]


def closed_loop(t, x):
    # Wrap alpha to [-pi, pi]
    alpha_wrapped = (x[1] + np.pi) % (2 * np.pi) - np.pi

    x_ctrl = np.array([x[0], alpha_wrapped, x[2], x[3]])

    # Only activate LQR when pendulum is near upright (|alpha| < 40°)
    if abs(alpha_wrapped) < np.radians(40):
        u = float((-K @ x_ctrl).flatten()[0])
        u = np.clip(u, -0.5, 0.5)   # motor torque saturation [N·m]
    else:
        u = 0.0  # swing-up not implemented here

    return furuta_dynamics(t, x, u)


# ══════════════════════════════════════════════
# SIMULATE
# ══════════════════════════════════════════════
t_span = (0, 5.0)
t_eval = np.linspace(*t_span, 2000)

# Initial condition: pendulum slightly off upright
x0 = [0.0,           # theta = 0
      np.radians(15), # alpha = 15° off upright
      0.0,            # dtheta = 0
      0.0]            # dalpha = 0

sol = solve_ivp(closed_loop, t_span, x0,
                t_eval=t_eval, method='RK45',
                max_step=0.002, rtol=1e-6)

theta = sol.y[0]
alpha = sol.y[1]
dtheta = sol.y[2]
dalpha  = sol.y[3]
t = sol.t

# Control input at each timestep
u_hist = np.array([
    float((-K @ np.array([(theta[i]), ((alpha[i]+np.pi)%(2*np.pi)-np.pi),
                dtheta[i], dalpha[i]])).flatten()[0])
    for i in range(len(t))
])
u_hist = np.clip(u_hist, -0.5, 0.5)

# ══════════════════════════════════════════════
# PLOT STATE TRAJECTORIES
# ══════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Furuta Pendulum — LQR Control", fontsize=14)

axes[0].plot(t, np.degrees(theta), color='steelblue', linewidth=1.5)
axes[0].set_ylabel("θ — Arm angle [°]")
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, np.degrees(alpha), color='coral', linewidth=1.5)
axes[1].set_ylabel("α — Pendulum angle [°]")
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].fill_between(t, -40, 40, alpha=0.05, color='green',
                     label='LQR active zone')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, u_hist * 1000, color='purple', linewidth=1.5)
axes[2].set_ylabel("Control torque [mN·m]")
axes[2].set_xlabel("Time [s]")
axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_states = os.path.join(OUT_DIR, "furuta_states.png")
plt.savefig(out_states, dpi=150, bbox_inches='tight')
print(f"State plot saved: {out_states}")

# ══════════════════════════════════════════════
# 3D ANIMATION
# ══════════════════════════════════════════════
fig_anim = plt.figure(figsize=(9, 7))
ax3d = fig_anim.add_subplot(111, projection='3d')

# Downsample for animation
n_frames = 200
idx = np.linspace(0, len(t)-1, n_frames, dtype=int)
theta_a = theta[idx]
alpha_a = alpha[idx]
t_a = t[idx]

def get_positions(th, al):
    # Motor shaft at origin
    # Arm tip (pendulum pivot)
    px = L1 * np.cos(th)
    py = L1 * np.sin(th)
    pz = 0.0
    # Pendulum tip
    # alpha=0 → upright (positive Z)
    tx = px - L2 * np.sin(al) * np.cos(th)
    ty = py - L2 * np.sin(al) * np.sin(th)
    tz = pz + L2 * np.cos(al)
    return (px, py, pz), (tx, ty, tz)

arm_line,     = ax3d.plot([], [], [], 'o-',
                          color='steelblue', linewidth=3,
                          markersize=6, label='Arm')
pend_line,    = ax3d.plot([], [], [], 'o-',
                          color='coral', linewidth=3,
                          markersize=8, label='Pendulum')
trace_line,   = ax3d.plot([], [], [], '-',
                          color='coral', linewidth=0.8,
                          alpha=0.4)
time_text = ax3d.text2D(0.02, 0.95, '', transform=ax3d.transAxes,
                         fontsize=10)
angle_text = ax3d.text2D(0.02, 0.88, '', transform=ax3d.transAxes,
                          fontsize=9, color='coral')

ax3d.set_xlim(-0.15, 0.15)
ax3d.set_ylim(-0.15, 0.15)
ax3d.set_zlim(-0.05, 0.15)
ax3d.set_xlabel('X [m]')
ax3d.set_ylabel('Y [m]')
ax3d.set_zlabel('Z [m]')
ax3d.set_title('Furuta Pendulum — 3D Animation')
ax3d.legend(loc='upper right', fontsize=9)

ax3d.plot([0], [0], [0], 'k+', markersize=12, markeredgewidth=2)

trace_x, trace_y, trace_z = [], [], []
TRACE_LEN = 40

def init():
    arm_line.set_data([], [])
    arm_line.set_3d_properties([])
    pend_line.set_data([], [])
    pend_line.set_3d_properties([])
    trace_line.set_data([], [])
    trace_line.set_3d_properties([])
    return arm_line, pend_line, trace_line, time_text, angle_text

def animate(i):
    th = theta_a[i]
    al = alpha_a[i]
    pivot, tip = get_positions(th, al)

    arm_line.set_data([0, pivot[0]], [0, pivot[1]])
    arm_line.set_3d_properties([0, pivot[2]])

    pend_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
    pend_line.set_3d_properties([pivot[2], tip[2]])

    trace_x.append(tip[0])
    trace_y.append(tip[1])
    trace_z.append(tip[2])
    if len(trace_x) > TRACE_LEN:
        trace_x.pop(0); trace_y.pop(0); trace_z.pop(0)
    trace_line.set_data(trace_x, trace_y)
    trace_line.set_3d_properties(trace_z)

    time_text.set_text(f't = {t_a[i]:.2f} s')
    angle_text.set_text(f'α = {np.degrees(al):.1f}°')

    return arm_line, pend_line, trace_line, time_text, angle_text

anim = animation.FuncAnimation(
    fig_anim, animate, init_func=init,
    frames=n_frames, interval=25, blit=True
)

out_anim = os.path.join(OUT_DIR, "furuta_animation.gif")
anim.save(out_anim, writer='pillow', fps=30, dpi=100)
print(f"Animation saved: {out_anim}")

plt.show()
print("Done!")
