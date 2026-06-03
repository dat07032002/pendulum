import mujoco
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("furuta_pendulum.xml")
dt = model.opt.timestep

def run(duration, q0, lock_shoulder=True, ctrl=0.0):
    data = mujoco.MjData(model)
    data.qpos[:len(q0)] = q0
    mujoco.mj_forward(model, data)
    times, shoulder, elbow = [], [], []
    for i in range(int(duration / dt)):
        data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)
        if lock_shoulder:
            data.qvel[0] = 0   # freeze arm
            data.qacc[0] = 0
        times.append(i * dt)
        shoulder.append(np.degrees(data.qpos[0]))
        elbow.append(np.degrees(data.qpos[1]))
    return np.array(times), np.array(shoulder), np.array(elbow)

# ── Find equilibrium ─────────────────────────────────────────────────────────
t, _, el = run(5.0, [0, 0], lock_shoulder=True)
equilibrium = el[-1]
print(f"Equilibrium angle (pendulum at rest): {equilibrium:.2f} deg")

# ── Test 1: Period ───────────────────────────────────────────────────────────
# start 15 deg from equilibrium, lock shoulder, measure period
start_angle = np.radians(equilibrium + 15)
t, _, el = run(5.0, [0, start_angle], lock_shoulder=True)

crossings = []
for i in range(1, len(el)):
    if (el[i-1] - equilibrium) * (el[i] - equilibrium) < 0:
        crossings.append(t[i])

if len(crossings) >= 2:
    sim_period = 2 * (crossings[1] - crossings[0])
    print(f"Test 1 — Sim period: {sim_period:.3f} s")
else:
    sim_period = None
    print(f"Test 1 — No crossings found. Angle range: {el.min():.1f} to {el.max():.1f} deg")

# Theoretical period
pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_rod")
jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
data_tmp = mujoco.MjData(model)
data_tmp.qpos[1] = np.radians(equilibrium)
mujoco.mj_forward(model, data_tmp)
pivot = data_tmp.xanchor[jid]
com = data_tmp.xipos[pid]
L = np.linalg.norm(com - pivot)
T_theory = 2 * np.pi * np.sqrt(L / 9.81)
print(f"Test 1 — L (CoM to pivot): {L*100:.2f} cm")
print(f"Test 1 — Theoretical period: {T_theory:.3f} s")

# ── Test 2: Visual swing ─────────────────────────────────────────────────────
t2, sh2, el2 = run(4.0, [0, np.radians(equilibrium + 20)], lock_shoulder=True)
print(f"Test 2 — Angle range: {el2.min():.1f} to {el2.max():.1f} deg (should oscillate around {equilibrium:.0f} deg)")

# ── Test 3: Energy conservation ─────────────────────────────────────────────
drop = 40.0
t3, _, el3 = run(4.0, [0, np.radians(equilibrium + drop)], lock_shoulder=True)
# find max excursion on the return swing (other side of equilibrium)
crossed = False
max_return = 0.0
for a in el3:
    if not crossed and a < equilibrium:
        crossed = True
    if crossed and a > equilibrium:
        max_return = max(max_return, a - equilibrium)

print(f"Test 3 — Drop: {drop:.0f} deg from equilibrium")
print(f"Test 3 — Return: {max_return:.1f} deg ({max_return/drop*100:.0f}% energy retained)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(t, el, 'r')
axes[0].axhline(equilibrium, color='gray', linestyle='--', label=f'equilibrium {equilibrium:.0f} deg')
title1 = f"Test 1: Period | Sim: {sim_period:.3f}s  Theory: {T_theory:.3f}s" if sim_period else f"Test 1: No oscillation detected (L={L*100:.1f}cm)"
axes[0].set_title(title1)
axes[0].set_ylabel("Pendulum angle (deg)")
axes[0].legend()

axes[1].plot(t2, el2, 'b')
axes[1].axhline(equilibrium, color='gray', linestyle='--', label=f'equilibrium')
axes[1].set_title("Test 2: Visual swing (shoulder locked)")
axes[1].set_ylabel("Pendulum angle (deg)")
axes[1].legend()

axes[2].plot(t3, el3, 'g')
axes[2].axhline(equilibrium, color='gray', linestyle='--', label='equilibrium')
axes[2].set_title(f"Test 3: Energy conservation | Drop: {drop:.0f} deg  Return: {max_return:.1f} deg ({max_return/drop*100:.0f}%)")
axes[2].set_ylabel("Pendulum angle (deg)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig("sanity_check.png", dpi=150)
plt.show()
print("Saved sanity_check.png")
