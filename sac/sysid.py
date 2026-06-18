"""
sac/sysid.py — hardware system identification for the Furuta pendulum.

Runs three experiments with the pendulum hanging (theta ~= -180 deg):

  Phase 1 — terminal-velocity sweep
    Apply step u, wait for steady state, record phi_dot_ss.
    Fit: phi_dot_ss = (gear / damping) * u  (linear regression through origin)

  Phase 2 — deceleration
    Spin arm to near-terminal velocity, cut motor to zero, record phi_dot decay.
    Fit: phi_dot(t) = A * exp(-t / tau),  tau = I_arm / damping

  Phase 3 — transport delay
    Step u=0 -> u=U_STEP, measure time until first detectable phi_dot rise.

From these three numbers:
    damping = I_arm / tau
    gear    = (gear / damping) * damping

Prints fitted values and suggested XML + furuta_env.py edits.

Usage:
    python sac/sysid.py --port COM5
    python sac/sysid.py --port COM5 --phases 1,2     # skip delay test
    python sac/sysid.py --port COM5 --arm-inertia 1.8e-4
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    raise SystemExit("pyserial not installed:  pip install pyserial")

try:
    from scipy.optimize import curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Phase 1: terminal velocity sweep
U_SWEEP        = [0.10, 0.15, 0.20, 0.25, 0.30]
STEP_SECONDS   = 1.5    # run duration per u value (s)

# Phase 2: deceleration
# Use a low u so the arm stays well inside the backstop during spin-up,
# leaving room to coast to a stop.  Nidec is fast: 0.15 covers ~30 deg in 0.5s.
U_SPINUP         = 0.15
SPINUP_SECONDS   = 0.5
DECAY_SECONDS    = 1.0   # longer decay window at lower speed

# Phase 3: transport delay
U_DELAY_STEP     = 0.30
DELAY_THRESHOLD  = 0.05   # rad/s phi_dot threshold to declare "motion detected"
DELAY_TIMEOUT    = 0.5    # s to wait for motion

# Safety backstop for Phase 1 runs (firmware backstop is 120 deg).
# Phase 2 uses a tighter limit so the arm has room to coast after cut.
PHI_BACKSTOP_DEG        = 80.0   # Phase 1 & 3
PHI_BACKSTOP_DEG_DECAY  = 40.0   # Phase 2 spin-up: stop early, leave coasting room

# Nominal arm inertia used when the user does not supply --arm-inertia.
# Derived from the XML: arm mass=0.030 kg, arm length ~0.10 m,
# rotating about shoulder pivot -> I ~= 1/3 * m * L^2 = 1.0e-4 kg*m^2.
# Motor rotor inertia is unknown; use --arm-inertia to override.
ARM_INERTIA_NOMINAL = 1.00e-4  # kg·m²

OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------

def _parse_obs(line: str) -> np.ndarray | None:
    m = OBS_RE.search(line)
    if not m:
        return None
    parts = m.group(1).split(",")
    if len(parts) != 5:
        return None
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        return None


class HW:
    """Thin serial wrapper matching the nidec_policy firmware protocol."""

    def __init__(self, port: str, baud: int = 921600):
        self._s = serial.Serial(port, baud, timeout=0.005)
        time.sleep(2.0)          # ESP32 reboots on DTR toggle at serial open
        self._s.reset_input_buffer()
        self._cmd("u 0.0")

    def _cmd(self, s: str) -> None:
        self._s.write((s + "\n").encode("ascii"))
        self._s.flush()

    def u(self, val: float) -> None:
        self._cmd(f"u {val:.5f}")

    def stop(self) -> None:
        self._cmd("s")

    def zero(self) -> None:
        self._cmd("z")
        time.sleep(0.1)
        self._s.reset_input_buffer()

    def drain(self, seconds: float,
              backstop_deg: float = PHI_BACKSTOP_DEG) -> list[np.ndarray]:
        """Collect obs lines for `seconds`; stop early on phi backstop."""
        out: list[np.ndarray] = []
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            raw = self._s.readline()
            if not raw:
                continue
            obs = _parse_obs(raw.decode("utf-8", errors="replace"))
            if obs is None:
                continue
            out.append(obs)
            if abs(np.degrees(float(obs[3]))) > backstop_deg:
                print(f"    backstop: phi={np.degrees(obs[3]):+.1f} deg, stopping early")
                break
        return out

    def wait_still(self, phi_tol_deg: float = 15.0,
                   phi_dot_tol: float = 0.3,
                   timeout: float = 20.0) -> bool:
        """Block until |phi| < tol and arm is slow.  Returns False on timeout."""
        self.stop()
        deadline = time.perf_counter() + timeout
        settled: float | None = None
        while time.perf_counter() < deadline:
            raw = self._s.readline()
            if not raw:
                continue
            obs = _parse_obs(raw.decode("utf-8", errors="replace"))
            if obs is None:
                continue
            if (abs(np.degrees(float(obs[3]))) < phi_tol_deg
                    and abs(float(obs[4])) < phi_dot_tol):
                if settled is None:
                    settled = time.perf_counter()
                elif time.perf_counter() - settled >= 0.4:
                    return True
            else:
                settled = None
        return False

    def close(self) -> None:
        try:
            self.stop()
            self._s.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1 — terminal velocity
# ---------------------------------------------------------------------------

def phase1(hw: HW, u_values: list[float]) -> dict[float, float]:
    """Returns {u: phi_dot_ss} measured at steady state."""
    print("\n=== Phase 1: Terminal Velocity Sweep ===")
    print("Keep the pendulum hanging.  You will center the arm before each run.")
    results: dict[float, float] = {}

    for u in u_values:
        if not hw.wait_still():
            print(f"  WARNING: arm didn't settle; skipping u={u:.2f}")
            continue
        input(f"\n  Center the arm, then press Enter  [u={u:.2f}]...")
        hw.zero()
        hw.u(u)
        obs_list = hw.drain(STEP_SECONDS)
        hw.stop()

        if not obs_list:
            print(f"  No obs received for u={u:.2f}")
            continue

        # median of the last 40% of samples = steady state
        tail = obs_list[int(len(obs_list) * 0.6):]
        phi_dot_ss = float(np.median([abs(float(o[4])) for o in tail]))
        print(f"  u={u:.2f} -> phi_dot_ss = {phi_dot_ss:.4f} rad/s  ({len(obs_list)} samples)")
        results[u] = phi_dot_ss

    return results


# ---------------------------------------------------------------------------
# Phase 2 — deceleration
# ---------------------------------------------------------------------------

def _fit_exp_decay(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y(t) = A * exp(-t / tau).  Returns (A, tau)."""
    if _SCIPY:
        popt, _ = curve_fit(
            lambda t, A, tau: A * np.exp(-t / tau),
            t, y, p0=[y[0], 0.2],
            bounds=([0, 0.005], [np.inf, 5.0]),
            maxfev=2000,
        )
        return float(popt[0]), float(popt[1])
    # Fallback: log-linear regression
    valid = y > 0.01
    if valid.sum() < 3:
        raise ValueError("not enough non-zero samples for log-linear fit")
    coeffs = np.polyfit(t[valid], np.log(y[valid]), 1)
    tau = -1.0 / coeffs[0]
    A = float(np.exp(coeffs[1]))
    return A, float(tau)


def phase2(hw: HW) -> float:
    """Spin up then cut motor; fit exponential phi_dot decay.  Returns tau (s)."""
    print("\n=== Phase 2: Deceleration ===")
    print("Keep the pendulum hanging.")

    if not hw.wait_still():
        print("  WARNING: arm didn't settle before spin-up")
    input(f"\n  Center the arm, then press Enter  [spin-up u={U_SPINUP:.2f}]...")

    hw.zero()
    hw.u(U_SPINUP)
    # Tight backstop: stop spin-up early so the arm has room to coast.
    hw.drain(SPINUP_SECONDS, backstop_deg=PHI_BACKSTOP_DEG_DECAY)
    # Cut motor to zero (active zero-speed setpoint, keeps BRAKE_PIN HIGH)
    # "u 0" rather than "s" so electromagnetic braking matches the policy's
    # behaviour when it outputs a near-zero action.
    hw.u(0.0)
    decay_obs = hw.drain(DECAY_SECONDS, backstop_deg=PHI_BACKSTOP_DEG_DECAY * 3)
    hw.stop()

    if len(decay_obs) < 8:
        print("  Not enough decay samples; skipping phase 2")
        return float("nan")

    phi_dots = np.array([abs(float(o[4])) for o in decay_obs])
    n = len(phi_dots)
    dt_est = DECAY_SECONDS / n
    t = np.arange(n, dtype=float) * dt_est

    try:
        A, tau = _fit_exp_decay(t, phi_dots)
    except Exception as exc:
        print(f"  Exponential fit failed: {exc}")
        return float("nan")

    print(f"  Decay fit: A={A:.4f} rad/s,  tau={tau:.4f} s  ({n} samples)")
    print(f"  Note: phi_dot_filt (alpha=0.85 @ ~200 Hz) adds ~33 ms lag;")
    print(f"        fitted tau may be slightly over-estimated.")
    return tau


# ---------------------------------------------------------------------------
# Phase 3 — transport delay
# ---------------------------------------------------------------------------

def phase3(hw: HW) -> float:
    """Step u=0 -> U_DELAY_STEP; return time-to-first-motion in ms."""
    print("\n=== Phase 3: Transport Delay ===")
    print("Keep the pendulum hanging.")

    if not hw.wait_still():
        print("  WARNING: arm didn't settle")
    input(f"\n  Center the arm, then press Enter  [step u={U_DELAY_STEP:.2f}]...")

    hw._s.reset_input_buffer()
    t0 = time.perf_counter()
    hw.u(U_DELAY_STEP)
    deadline = time.perf_counter() + DELAY_TIMEOUT
    delay_ms = float("nan")

    while time.perf_counter() < deadline:
        raw = hw._s.readline()
        if not raw:
            continue
        obs = _parse_obs(raw.decode("utf-8", errors="replace"))
        if obs is None:
            continue
        if abs(float(obs[4])) > DELAY_THRESHOLD:
            delay_ms = (time.perf_counter() - t0) * 1000.0
            break

    hw.stop()

    if not np.isnan(delay_ms):
        print(f"  Transport delay ≈ {delay_ms:.1f} ms  "
              f"(threshold {DELAY_THRESHOLD} rad/s)")
    else:
        print(f"  No motion above {DELAY_THRESHOLD} rad/s within {DELAY_TIMEOUT} s")
        print("  Check that phi_dot is being read correctly.")

    return delay_ms


# ---------------------------------------------------------------------------
# Parameter fitting and reporting
# ---------------------------------------------------------------------------

def fit_and_report(
    terminal: dict[float, float],
    tau: float,
    arm_inertia: float,
    delay_ms: float,
) -> dict:
    """Compute gear and damping; print suggested XML edits."""

    print("\n=== Fitting Parameters ===")

    # --- gear/damping ratio from terminal velocity ---
    u_arr = np.array(sorted(terminal.keys()))
    v_arr = np.array([terminal[u] for u in u_arr])
    # Linear regression through origin: phi_dot_ss = slope * u
    slope = float(np.dot(u_arr, v_arr) / np.dot(u_arr, u_arr))
    residuals = v_arr - slope * u_arr
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    print(f"  Terminal velocity slope (gear/damping) = {slope:.4f} rad/s per unit u")
    print(f"  Fit RMSE = {rmse:.4f} rad/s")

    # --- damping and gear from tau ---
    if np.isnan(tau) or tau <= 0:
        print("  tau is invalid; cannot compute absolute gear/damping.")
        print("  Only the gear/damping ratio is available.")
        gear, damping = float("nan"), float("nan")
    else:
        damping = arm_inertia / tau
        gear    = slope * damping
        print(f"  tau = {tau:.4f} s,  I_arm = {arm_inertia:.2e} kg·m²")
        print(f"  damping = I_arm / tau = {damping:.5f} N·m·s/rad")
        print(f"  gear    = slope * damping = {gear:.4f} N·m")

    # --- transport delay ---
    delay_steps = None
    if not np.isnan(delay_ms):
        delay_steps = max(0, round(delay_ms / 10.0))

    # --- report ---
    print()
    print("=" * 55)
    print("SUGGESTED EDITS")
    print("=" * 55)
    if not np.isnan(gear):
        print(f"\nfuruta_pendulum.xml  (sac/ or pendulum/):")
        print(f'  <motor name="shoulder_motor" joint="shoulder"')
        print(f'         gear="{gear:.4f}" ctrlrange="-1 1"/>')
        print(f'  <joint name="shoulder" ... damping="{damping:.5f}" .../>')
    else:
        print(f"\nOnly ratio available.  Set gear=0.132 and adjust damping so that")
        print(f"  gear/damping = {slope:.4f}  =>  damping = {0.132/slope:.5f} N·m·s/rad")
        damping = 0.132 / slope
        gear    = 0.132

    if delay_steps is not None and delay_steps >= 1:
        print(f"\nsac/furuta_env.py  (if you add a max_delay parameter):")
        print(f"  self._max_delay = {delay_steps}  (measured ~{delay_ms:.0f} ms at 100 Hz)")
    elif delay_steps == 0:
        print(f"\n  Transport delay < 10 ms; current _max_delay=1 is conservative (fine).")

    print(f"\nDomain randomization ranges to consider:")
    if not np.isnan(gear):
        print(f"  motor_gear_dr_range    : ±0.05  (gear     = {gear:.4f})")
        print(f"  shoulder_damp_dr_range : ±0.10  (damping  = {damping:.5f})")
    print()

    return {
        "gear_over_damping_slope": slope,
        "slope_rmse": rmse,
        "tau_s": None if np.isnan(tau) else tau,
        "arm_inertia_kgm2": arm_inertia,
        "fitted_gear": None if np.isnan(gear) else gear,
        "fitted_damping": None if np.isnan(damping) else damping,
        "transport_delay_ms": None if np.isnan(delay_ms) else delay_ms,
        "transport_delay_steps_at_100Hz": delay_steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Furuta pendulum hardware system identification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",  default="COM5")
    parser.add_argument("--baud",  type=int, default=921600)
    parser.add_argument("--arm-inertia", type=float, default=ARM_INERTIA_NOMINAL,
                        metavar="KGM2",
                        help="Arm rotational inertia in kg·m² "
                             f"(default: {ARM_INERTIA_NOMINAL:.2e} from XML masses)")
    parser.add_argument("--phases", default="1,2,3",
                        help="Comma-separated phases to run, e.g. 1,2 to skip delay")
    parser.add_argument("--u-sweep", default=None,
                        help="Override u values for phase 1, e.g. 0.1,0.2,0.3")
    parser.add_argument("--out", default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    phases = {int(p.strip()) for p in args.phases.split(",")}
    u_values = ([float(x) for x in args.u_sweep.split(",")]
                if args.u_sweep else U_SWEEP)

    print("=== Furuta Pendulum System Identification ===")
    print(f"Port: {args.port}  Baud: {args.baud}")
    print(f"I_arm: {args.arm_inertia:.2e} kg·m²  (use --arm-inertia to override)")
    print(f"Phases: {sorted(phases)}")
    print()
    print("IMPORTANT: keep the pendulum hanging throughout (theta ~= -180 deg).")
    print()

    hw = HW(args.port, args.baud)
    terminal: dict[float, float] = {}
    tau      = float("nan")
    delay_ms = float("nan")

    try:
        if 1 in phases:
            terminal = phase1(hw, u_values)

        if 2 in phases:
            tau = phase2(hw)

        if 3 in phases:
            delay_ms = phase3(hw)

        if terminal:
            results = fit_and_report(terminal, tau, args.arm_inertia, delay_ms)
        else:
            print("\nNo terminal-velocity data collected; nothing to fit.")
            results = {}

        if args.out:
            results["terminal_velocity"] = {str(k): v for k, v in terminal.items()}
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Results saved -> {args.out}")

    finally:
        hw.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
