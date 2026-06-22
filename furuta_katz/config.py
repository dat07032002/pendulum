"""
config.py — one home for every runtime constant.

Model constants live in plant.py (geometry-derived, validated in sim) and are
re-exported here for convenience. The motor-calibration numbers (V_MAX, U_MIN)
are measured by calibrate.py and persisted to calibration.json; hw_env.py refuses
to drive the motor until they exist, so a balance run can never use a guessed scale.
"""
from __future__ import annotations

import json
import os

import plant

# --- serial link to the ESP32 ---
PORT = "COM5"
BAUD = 921600
CONTROL_DT = plant.DT          # 0.006 s (166 Hz measured); the run loop paces to this

# --- model constants (defined in plant.py; mirrored here for one-stop reading) ---
ALPHA = plant.ALPHA
BETA = plant.BETA
L_ROD = plant.L_ROD
L_ARM = plant.L_ARM

# --- cable-wrap / safety limits ---
PHI_LIMIT_DEG = 120.0          # hard cable limit (matches the firmware backstop)
PHI_SOFT_DEG = 90.0            # controllers should stay inside this

# --- motor calibration (filled in by calibrate.py) -----------------------
# None until measured. is_calibrated() gates hw_env from driving.
V_MAX = None                   # arm speed [rad/s] at |u|=1  -> u = clip(v_cmd / V_MAX, +-1)
U_MIN = None                   # smallest |u| that moves the arm (speed-loop stall floor)

_CAL_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


def load_calibration() -> tuple[float | None, float | None]:
    """Load V_MAX / U_MIN from calibration.json if present (called on import)."""
    global V_MAX, U_MIN
    if os.path.exists(_CAL_FILE):
        with open(_CAL_FILE) as f:
            d = json.load(f)
        V_MAX = d.get("V_MAX", V_MAX)
        U_MIN = d.get("U_MIN", U_MIN)
    return V_MAX, U_MIN


def save_calibration(v_max: float, u_min: float, extra: dict | None = None) -> str:
    d = {"V_MAX": v_max, "U_MIN": u_min}
    if extra:
        d.update(extra)
    with open(_CAL_FILE, "w") as f:
        json.dump(d, f, indent=2)
    return _CAL_FILE


def is_calibrated() -> bool:
    return V_MAX is not None and U_MIN is not None


load_calibration()


if __name__ == "__main__":
    print(f"PORT={PORT}  BAUD={BAUD}  CONTROL_DT={CONTROL_DT:.5f} s")
    print(f"ALPHA={ALPHA:.2f}  BETA={BETA:.3f}  L_ROD={L_ROD}  L_ARM={L_ARM}")
    print(f"PHI_LIMIT={PHI_LIMIT_DEG} deg  PHI_SOFT={PHI_SOFT_DEG} deg")
    print(f"V_MAX={V_MAX}  U_MIN={U_MIN}  calibrated={is_calibrated()}")
    if not is_calibrated():
        print("-> run calibrate.py (motor moves) to measure V_MAX and U_MIN.")
