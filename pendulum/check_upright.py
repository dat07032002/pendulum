"""
Steps through key elbow angles and holds each for 3 seconds.
Watch the viewer and note which position looks visually upright.
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

XML_PATH = Path(__file__).with_name("furuta_pendulum.xml")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data  = mujoco.MjData(model)

angles_deg = [0, 5, -5, 10, -10, 180, -180]

with mujoco.viewer.launch_passive(model, data) as viewer:
    for deg in angles_deg:
        rad = np.radians(deg)
        mujoco.mj_resetData(model, data)
        data.qpos[0] = 0.0   # shoulder centred
        data.qpos[1] = rad
        mujoco.mj_forward(model, data)
        viewer.sync()
        input(f"\nqpos[1] = {rad:+.4f} rad  ({deg:+.1f} deg) — upright? Press Enter for next...")
        if not viewer.is_running():
            break

print("\nDone. Close the viewer window.")
