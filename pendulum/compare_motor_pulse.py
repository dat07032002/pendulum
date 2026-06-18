"""
Compare simulated and real shoulder response for a short motor pulse.

The ESP32 firmware currently applies deadband compensation, so a firmware
command u=0.05 maps to approximately:
    pwm = 50 + 0.05 * (255 - 50) = 60.25
    effective raw command ~= 60.25 / 255 = 0.236

By default this script simulates that effective raw command for 0.3 s using
the current MuJoCo XML.  Optionally, it can also run the real hardware pulse
over serial and compare final phi.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).with_name("furuta_pendulum.xml")
OBS_RE = re.compile(r"obs=\[([^\]]+)\]")


def compensated_to_raw_pwm_command(u: float, deadband_pwm: int = 50) -> float:
    if abs(u) < 1e-12:
        return 0.0
    mag = min(abs(u), 1.0)
    pwm = deadband_pwm + mag * (255 - deadband_pwm)
    return float(np.sign(u) * pwm / 255.0)


def simulate_pulse(action: float, duration: float, settle: float = 0.0) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    start_phi = float(data.qpos[0])
    peak_phi_dot = 0.0

    pulse_steps = round(duration / model.opt.timestep)
    settle_steps = round(settle / model.opt.timestep)

    for _ in range(pulse_steps):
        data.ctrl[0] = action
        mujoco.mj_step(model, data)
        peak_phi_dot = max(peak_phi_dot, abs(float(data.qvel[0])))

    stop_phi = float(data.qpos[0])

    for _ in range(settle_steps):
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        peak_phi_dot = max(peak_phi_dot, abs(float(data.qvel[0])))

    final_phi = float(data.qpos[0])

    return {
        "start_phi_rad": start_phi,
        "stop_phi_rad": stop_phi,
        "final_phi_rad": final_phi,
        "delta_at_stop_rad": stop_phi - start_phi,
        "delta_final_rad": final_phi - start_phi,
        "peak_phi_dot_rad_s": peak_phi_dot,
        "gear": float(model.actuator_gear[0, 0]),
    }


def parse_obs(line: str) -> list[float] | None:
    match = OBS_RE.search(line)
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) != 5:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def run_hardware_pulse(port_name: str, u: float, duration: float, post_read: float) -> dict[str, float]:
    try:
        import serial
    except ImportError:
        print("Missing dependency: pyserial. Install with: python -m pip install pyserial")
        raise

    def send_u(port, value: float) -> None:
        port.write(f"u {value:.5f}\n".encode("ascii"))
        port.flush()

    phis: list[float] = []
    phi_dots: list[float] = []
    pulse_start: float | None = None
    stopped = False

    with serial.Serial(port_name, 115200, timeout=0.02) as port:
        time.sleep(2.0)
        port.reset_input_buffer()
        send_u(port, 0.0)
        start = time.perf_counter()
        command_start = start + 0.5
        end = command_start + duration + post_read

        while time.perf_counter() < end:
            now = time.perf_counter()
            if pulse_start is None and now >= command_start:
                send_u(port, u)
                pulse_start = now
            if pulse_start is not None and not stopped and now - pulse_start >= duration:
                send_u(port, 0.0)
                stopped = True

            raw = port.readline()
            if not raw:
                continue
            obs = parse_obs(raw.decode("utf-8", errors="replace").strip())
            if obs is None:
                continue
            phis.append(obs[3])
            phi_dots.append(obs[4])

        send_u(port, 0.0)

    if not phis:
        raise RuntimeError("No observations received from hardware.")

    return {
        "start_phi_rad": phis[0],
        "final_phi_rad": phis[-1],
        "delta_final_rad": phis[-1] - phis[0],
        "peak_phi_dot_rad_s": max(abs(v) for v in phi_dots),
        "obs_count": float(len(phis)),
    }


def print_result(label: str, result: dict[str, float]) -> None:
    print(label)
    for key, value in result.items():
        if key.endswith("_rad"):
            print(f"  {key}: {value:+.5f} rad ({np.degrees(value):+.2f} deg)")
        elif key.endswith("_rad_s"):
            print(f"  {key}: {value:+.5f} rad/s ({np.degrees(value):+.2f} deg/s)")
        else:
            print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare sim and hardware motor pulse response.")
    parser.add_argument("--firmware-u", type=float, default=0.05, help="ESP32 firmware command u")
    parser.add_argument("--deadband-pwm", type=int, default=50, help="Firmware deadband compensation PWM")
    parser.add_argument("--duration", type=float, default=0.3, help="Pulse duration in seconds")
    parser.add_argument("--settle", type=float, default=1.0, help="Sim settle/read duration after stopping")
    parser.add_argument("--hardware", action="store_true", help="Also run the real ESP32 pulse")
    parser.add_argument("--port", default="COM5", help="Hardware serial port")
    args = parser.parse_args()

    sim_action = compensated_to_raw_pwm_command(args.firmware_u, args.deadband_pwm)
    print(f"Firmware u={args.firmware_u:+.3f} with deadband PWM {args.deadband_pwm}")
    print(f"Equivalent raw sim action ~= {sim_action:+.5f}")
    print()

    sim = simulate_pulse(sim_action, args.duration, args.settle)
    print_result("Simulation result:", sim)

    if args.hardware:
        print()
        hw = run_hardware_pulse(args.port, args.firmware_u, args.duration, args.settle)
        print_result("Hardware result:", hw)
        print()
        sim_deg = np.degrees(sim["delta_final_rad"])
        hw_deg = np.degrees(hw["delta_final_rad"])
        print(f"Final delta comparison: sim={sim_deg:+.2f} deg, hardware={hw_deg:+.2f} deg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
