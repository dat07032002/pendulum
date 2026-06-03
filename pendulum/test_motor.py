import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

XML_PATH = Path(__file__).with_name("furuta_pendulum.xml")
UPRIGHT_Q = 0.0
DOWN_Q = -np.pi

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
dt = model.opt.timestep

def run_sim(duration, ctrl_fn, q0=None):
    data = mujoco.MjData(model)
    data.qpos[:] = q0 if q0 is not None else [0.0, UPRIGHT_Q]
    steps = int(duration / dt)
    time, shoulder, elbow, vel = [], [], [], []
    ctrl_log = []
    for i in range(steps):
        t = i * dt
        data.ctrl[0] = ctrl_fn(t, data)
        mujoco.mj_step(model, data)
        time.append(t)
        shoulder.append(np.degrees(data.qpos[0]))
        elbow.append(np.degrees(data.qpos[1]))
        vel.append(np.degrees(data.qvel[0]))
        ctrl_log.append(data.ctrl[0])
    return np.array(time), np.array(shoulder), np.array(elbow), np.array(vel), np.array(ctrl_log)

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
fig.suptitle("Furuta Pendulum Motor Tests", fontsize=14)

# --- Test 1: Static torque ---
t, sh, el, v, c = run_sim(3.0, lambda t, d: 1.0)
axes[0,0].plot(t, sh, 'b', label='shoulder')
axes[0,0].plot(t, el, 'r', label='elbow')
axes[0,0].set_title("Test 1: Static torque (ctrl=1.0)")
axes[0,0].set_ylabel("Angle (deg)")
axes[0,0].legend()
axes[0,1].plot(t, v, 'orange')
axes[0,1].set_title("Shoulder velocity")
axes[0,1].set_ylabel("deg/s")
print(f"Test 1 — Final shoulder angle: {sh[-1]:.1f} deg, Final velocity: {v[-1]:.1f} deg/s")

# --- Test 2: Step response ---
t, sh, el, v, c = run_sim(3.0, lambda t, d: 1.0 if t < 1.0 else 0.0)
axes[1,0].plot(t, sh, 'b', label='shoulder')
axes[1,0].plot(t, el, 'r', label='elbow')
axes[1,0].set_title("Test 2: Step response (ctrl=1 for 1s, then 0)")
axes[1,0].set_ylabel("Angle (deg)")
axes[1,0].legend()
axes[1,1].plot(t, v, 'orange')
axes[1,1].set_title("Shoulder velocity")
axes[1,1].set_ylabel("deg/s")
peak_vel = np.max(np.abs(v[:int(1.0/dt)]))
print(f"Test 2 — Peak velocity during step: {peak_vel:.1f} deg/s")

# --- Test 3: Motor hold (arm at 90 deg, pendulum upright) ---
q0 = [np.radians(90), UPRIGHT_Q]
def gravity_hold(t, data):
    if t < 0.5:
        return 0.0
    # try increasing ctrl until arm holds
    return 0.15
t, sh, el, v, c = run_sim(3.0, gravity_hold, q0=q0)
axes[2,0].plot(t, sh, 'b', label='shoulder')
axes[2,0].axhline(90, color='gray', linestyle='--', label='target 90 deg')
axes[2,0].set_title("Test 3: Motor hold (arm at 90 deg, ctrl=0.15)")
axes[2,0].set_ylabel("Angle (deg)")
axes[2,0].legend()
axes[2,1].plot(t, v, 'orange')
axes[2,1].set_title("Shoulder velocity")
axes[2,1].set_ylabel("deg/s")
print(f"Test 3 — Shoulder angle at end: {sh[-1]:.1f} deg (target: 90 deg)")

# --- Test 4: Swing-up pumping ---
def swing_pump(t, data):
    elbow_vel = data.qvel[1]
    return 1.0 if elbow_vel > 0 else -1.0
t, sh, el, v, c = run_sim(5.0, swing_pump, q0=[0.0, DOWN_Q])
axes[3,0].plot(t, el, 'r', label='elbow (pendulum)')
axes[3,0].set_title("Test 4: Swing-up pumping (energy injection)")
axes[3,0].set_ylabel("Pendulum angle (deg)")
axes[3,0].legend()
axes[3,1].plot(t, sh, 'b', label='shoulder (arm)')
axes[3,1].set_title("Arm angle during swing-up")
axes[3,1].set_ylabel("deg")
axes[3,1].legend()
max_swing = np.max(np.abs(el))
print(f"Test 4 — Max pendulum swing: {max_swing:.1f} deg")

for ax in axes.flat:
    ax.set_xlabel("Time (s)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("motor_tests.png", dpi=150)
plt.show()
print("\nSaved motor_tests.png")
