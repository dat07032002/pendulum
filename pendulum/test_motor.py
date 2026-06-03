import mujoco
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("furuta_pendulum.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep
duration = 3.0
steps = int(duration / dt)

time = np.zeros(steps)
shoulder_angle = np.zeros(steps)
shoulder_vel = np.zeros(steps)
ctrl_log = np.zeros(steps)

for i in range(steps):
    t = i * dt

    # step input: full torque for 1s, then zero
    if t < 1.0:
        data.ctrl[0] = 1.0
    else:
        data.ctrl[0] = 0.0

    mujoco.mj_step(model, data)

    time[i] = t
    shoulder_angle[i] = np.degrees(data.qpos[0])
    shoulder_vel[i] = np.degrees(data.qvel[0])
    ctrl_log[i] = data.ctrl[0]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time, ctrl_log, color="gray")
axes[0].set_ylabel("Control input")
axes[0].set_ylim(-1.2, 1.2)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_title("Motor step response")

axes[1].plot(time, shoulder_angle, color="blue")
axes[1].set_ylabel("Shoulder angle (deg)")
axes[1].axhline(0, color="black", linewidth=0.5)

axes[2].plot(time, shoulder_vel, color="orange")
axes[2].set_ylabel("Shoulder velocity (deg/s)")
axes[2].set_xlabel("Time (s)")
axes[2].axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("motor_step_response.png", dpi=150)
plt.show()
print("Saved motor_step_response.png")
