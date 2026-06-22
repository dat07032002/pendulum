"""
actuator_realism.py — find the REAL killer.

actuator_ceiling.py showed the deadzone+floor alone still holds in sim. So the hardware
failure comes from effects the sim omits. Add them one at a time and see which breaks the hold:

  1. velocity-loop LAG : the motor reaches a commanded speed over ~tau, not instantly
  2. theta_dot NOISE   : the AS5600 derivative is quantized/noisy (~0.3 rad/s steps)
  3. measurement LATENCY: control acts on a 1-2 cycle old state

Whatever breaks the hold is the thing to fix (and tells us if it's fixable in software).
"""
from __future__ import annotations

import numpy as np

import plant
from sim import FurutaSim
from balance import _wrap

K = np.array([-18.31577, 855.90381, -9.14419, 61.18292])


def quantize(v, v_deadzone=1.5, v_min=2.98, v_max=78.0):
    s = abs(v)
    if s < v_deadzone:
        return 0.0
    sg = 1.0 if v > 0 else -1.0
    if s < v_min:
        return sg * v_min
    if s > v_max:
        return sg * v_max
    return v


def run(tilt_deg, bvmax, tau, thd_noise, latency, kick=False, rough=False,
        seconds=6.0, a_max=400.0, seed=0):
    rng = np.random.default_rng(seed)
    sim = FurutaSim(phi_dot_max=78.0)
    x = sim.reset(theta0=np.deg2rad(tilt_deg))
    dt = sim.dt
    vcmd = 0.0
    best = streak = 0
    hist = []
    prev_v = 0.0
    dith = 0.0
    for _ in range(int(seconds / dt)):
        hist.append(x.copy())
        xm = hist[max(0, len(hist) - 1 - latency)]          # delayed measurement
        theta_m = _wrap(float(xm[plant.THETA]))
        thd_m = float(xm[plant.THETA_DOT]) + rng.normal(0, thd_noise)   # noisy theta_dot
        xv = np.array([xm[plant.PHI], theta_m, xm[plant.PHI_DOT], thd_m])
        a = float(np.clip(-(K @ xv), -a_max, a_max))
        vcmd = float(np.clip(vcmd + a * dt, -bvmax, bvmax))
        v_target = quantize(vcmd)
        if rough and 0.0 < abs(vcmd) < 2.98:
            # firmware reality: sub-floor speeds are pulsed (0 or +-3), not smooth
            dith += abs(vcmd) / 2.98
            v_target = (np.sign(vcmd) * 2.98) if dith >= 1.0 else 0.0
            if dith >= 1.0:
                dith -= 1.0
        if kick and v_target != 0.0 and prev_v == 0.0:
            v_target = np.sign(v_target) * 16.0     # break-free kick spike (one cycle)
        prev_v = v_target
        a_real = (v_target - float(x[plant.PHI_DOT])) / max(tau, dt)     # velocity-loop lag
        x = sim.step(a_real)
        if abs(_wrap(float(x[plant.THETA]))) < np.deg2rad(15.0) and abs(float(x[plant.PHI])) < np.deg2rad(175.0):
            streak += 1; best = max(best, streak)
        else:
            streak = 0
            if abs(_wrap(float(x[plant.THETA]))) > np.deg2rad(60.0):
                break
    return best * dt


def best_bv(**kw):
    return max(run(bvmax=bv, **kw) for bv in (6, 9, 15, 30))


def main():
    dt = plant.DT
    print(f"Hold time [s] from a 5 deg catch (best over bvmax), control dt={dt*1000:.1f} ms:\n")
    rows = [
        ("baseline (instant, clean)",        dict(tau=dt,    thd_noise=0.0, latency=0)),
        ("velocity lag 100 ms",              dict(tau=0.100, thd_noise=0.0, latency=0)),
        ("velocity lag 200 ms",              dict(tau=0.200, thd_noise=0.0, latency=0)),
        ("theta_dot noise 1.0 rad/s",        dict(tau=dt,    thd_noise=1.0, latency=0)),
        ("latency 5 cycles (~30 ms)",        dict(tau=dt,    thd_noise=0.0, latency=5)),
        ("ROUGH sub-floor dither",           dict(tau=dt,    thd_noise=0.0, latency=0, rough=True)),
        ("KICK spikes (16 rad/s)",           dict(tau=dt,    thd_noise=0.0, latency=0, kick=True)),
        ("ROUGH + KICK + lag30 + noise.5",   dict(tau=0.030, thd_noise=0.5, latency=2, rough=True, kick=True)),
    ]
    for name, kw in rows:
        h = best_bv(tilt_deg=5, **kw)
        bar = "HELD" if h > 5.5 else ("falls" if h < 1.0 else "partial")
        print(f"  {name:<32} {h:5.2f} s   {bar}")
    print("\nWhatever first drops the hold toward 0 is the real killer.")


if __name__ == "__main__":
    main()
