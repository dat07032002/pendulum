"""run_steptest.py — send 'steptest', capture arm position vs time, and compute the
velocity-loop time constant (tau). MOVES THE ARM (positions, pre-rolls, then steps speed).

The sim (actuator_realism.py) showed tau >~100 ms is what makes balancing impossible.
This measures tau on the real motor. tau = time for the arm speed to cover 63% of the
LOW->target step. Prints steady speed, tau (63%), and t90."""
from __future__ import annotations
import re
import time
import numpy as np
import serial
import config

LINE = re.compile(r"t=(\d+)\s+tgt=(\d+)\s+phi=(-?[\d.]+)")
LOW = 3.5  # firmware pre-roll cruise speed [rad/s]


def main():
    ser = serial.Serial(config.PORT, config.BAUD, timeout=1.0)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.write(b"steptest\n"); ser.flush()
    print("sent steptest; arm repositions, pre-rolls, then steps (up to ~30s)...\n")
    data: dict[float, list[tuple[int, float]]] = {}
    deadline = time.time() + 70
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        s = raw.decode("utf-8", errors="replace").strip()
        m = LINE.search(s)
        if m:
            data.setdefault(float(m.group(2)), []).append((int(m.group(1)), float(m.group(3))))
        elif s == "DONE":
            break
        elif s.startswith("# steptest"):
            print(s)
    try:
        ser.write(b"s\n"); ser.flush()
    finally:
        ser.close()

    print("\n--- velocity-loop step response (fit to exact position) ---")
    from scipy.optimize import curve_fit

    # Arm position under a first-order velocity step from v1 to v2 with time constant tau:
    #   phi(t) = phi0 + v2*t - (v2 - v1)*tau*(1 - exp(-t/tau))
    # Fitting POSITION (exact to the encoder count) avoids the velocity-differencing noise.
    def pos_model(t, phi0, v1, v2, tau):
        tau = max(tau, 1e-3)
        return phi0 + v2 * t - (v2 - v1) * tau * (1.0 - np.exp(-t / tau))

    for tgt, samples in sorted(data.items()):
        if len(samples) < 8:
            print(f"target {tgt:.0f} rad/s: too few samples ({len(samples)})")
            continue
        a = np.array(sorted(samples), dtype=float)
        t = (a[:, 0] - a[0, 0]) / 1000.0
        phi = np.radians(a[:, 1])
        try:
            p0 = [phi[0], LOW, tgt, 0.05]
            popt, _ = curve_fit(pos_model, t, phi, p0=p0, maxfev=20000,
                                bounds=([-1e4, 0.0, 0.0, 0.002], [1e4, 20.0, 25.0, 1.0]))
            phi0, v1, v2, tau = popt
            resid = float(np.sqrt(np.mean((pos_model(t, *popt) - phi) ** 2)))
            print(f"target {tgt:4.0f} rad/s:  v1={v1:4.1f} -> v2={v2:4.1f} rad/s   "
                  f"tau={1000*tau:5.0f} ms   (fit RMS {np.degrees(resid):.1f} deg, n={len(t)})")
        except Exception as e:
            print(f"target {tgt:.0f} rad/s: fit failed ({e})")

    print("\nInterpretation:  tau < ~40 ms = fast enough to balance.")
    print("                 tau > ~80-100 ms = slow velocity loop = the balance killer.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"serial error: {e}\nClose balance_chip.py / Serial Monitor first (COM5 busy).")
