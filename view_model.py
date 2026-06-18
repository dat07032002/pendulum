"""
Interactive Furuta pendulum viewer.

Click on the viewer window to focus it, then use the terminal for key input:
  A  - rotate arm left
  D  - rotate arm right
  SPACE - stop arm
  Q  - quit

The pendulum swings freely under gravity.
"""
import mujoco
import mujoco.viewer
import time
import numpy as np
import threading
import msvcrt

m = mujoco.MjModel.from_xml_path("pendulum/furuta_pendulum.xml")
d = mujoco.MjData(m)

# Start with pendulum hanging
d.qpos[0] = 0.0
d.qpos[1] = -np.pi
mujoco.mj_forward(m, d)

arm_u = 0.0
running = True

def key_reader():
    global arm_u, running
    while running:
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            if key == b'a':
                arm_u = -0.5
                print("  arm <- left  (u=-0.5)")
            elif key == b'd':
                arm_u = 0.5
                print("  arm -> right (u=+0.5)")
            elif key == b' ':
                arm_u = 0.0
                print("  arm stopped")
            elif key == b'q':
                running = False
        time.sleep(0.02)

threading.Thread(target=key_reader, daemon=True).start()

print("=" * 40)
print("  A = rotate arm left")
print("  D = rotate arm right")
print("  SPACE = stop arm")
print("  Q = quit")
print("  (type in THIS terminal, not the viewer)")
print("=" * 40)

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.distance = 0.4
    v.cam.elevation = -20
    v.cam.azimuth = 135
    v.sync()

    step_t = time.perf_counter()
    while v.is_running() and running:
        d.ctrl[0] = arm_u
        mujoco.mj_step(m, d)
        v.sync()
        # real-time pacing
        step_t += m.opt.timestep
        sleep = step_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

running = False
print("Viewer closed.")
