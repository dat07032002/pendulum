"""
Run a trained PPO policy on real ESP32 observations with strict safety limits.

This is the first live policy bridge:
  - Reads obs=[cos(theta),sin(theta),theta_dot,phi,phi_dot] from ESP32.
  - Predicts PPO action.
  - Clamps action to a small range.
  - Sends the clamped action to ESP32 for a short duration.
  - Sends "u 0" on every exit path.

Start with the pendulum/arm mechanically safe and your hand near power.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover
    print("Missing dependency: pyserial")
    print("Install it with: python -m pip install pyserial")
    sys.exit(1)

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:  # pragma: no cover
    print("Missing dependency: stable-baselines3/gymnasium")
    print("Install your training dependencies before running this.")
    sys.exit(1)


OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
DEFAULT_MODEL_DIR = Path("runs") / "no_dr" / "20260611_clean_deadband0235"


class PolicySpaceEnv(gym.Env):
    metadata = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -55.0, -2.5, -20.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 55.0, 2.5, 20.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(5, dtype=np.float32), 0.0, False, False, {}


def parse_obs(line: str) -> np.ndarray | None:
    match = OBS_RE.search(line)
    if not match:
        return None

    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) != 5:
        return None

    try:
        return np.array([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None


def send_u(port: serial.Serial, u: float) -> None:
    port.write(f"u {u:.5f}\n".encode("ascii"))
    port.flush()


def load_policy(model_dir: Path) -> tuple[PPO, VecNormalize]:
    model_path = model_dir / "best_model.zip"
    norm_path = model_dir / "vec_normalize_best.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats: {norm_path}")

    model = PPO.load(model_path, device="cpu")
    dummy_env = DummyVecEnv([PolicySpaceEnv])
    vec_norm = VecNormalize.load(norm_path, dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Limited live PPO bridge for ESP32 hardware.")
    parser.add_argument("--port", default="COM5", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--seconds", type=float, default=0.3, help="Live policy duration")
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to send u 0 before enabling policy")
    parser.add_argument("--action-limit", type=float, default=0.25, help="Clamp action to +/- this value (applied after motor-scale)")
    parser.add_argument("--motor-scale", type=float, default=1.0, help="Multiply policy output by this before sending (e.g. 0.2 = 20% power)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Folder with best_model.zip and vec_normalize_best.pkl")
    parser.add_argument("--phi-limit-deg", type=float, default=90.0, help="Stop if abs(phi) exceeds this")
    parser.add_argument("--phi-dot-limit", type=float, default=8.0, help="Stop if abs(phi_dot) exceeds this (rad/s)")
    parser.add_argument("--timeout", type=float, default=0.2, help="Stop if no valid obs arrives within this many seconds")
    parser.add_argument("--startup-timeout", type=float, default=5.0, help="Stop if no first valid obs arrives within this many seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print clamped action but send u 0")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / model_dir

    action_limit = abs(args.action_limit)
    motor_scale = abs(args.motor_scale)
    phi_limit = np.radians(args.phi_limit_deg)
    phi_dot_limit = abs(args.phi_dot_limit)

    print(f"Loading policy from {model_dir}")
    model, vec_norm = load_policy(model_dir)
    print(f"Opening {args.port} at {args.baud} baud")
    print(
        "LIMITED LIVE TEST: "
        f"duration={args.seconds:.2f}s, action clamp=+/-{action_limit:.3f}, "
        f"phi limit=+/-{args.phi_limit_deg:.1f}deg, "
        f"phi_dot limit=+/-{phi_dot_limit:.1f}rad/s"
    )
    if args.dry_run:
        print("Dry-run flag is ON: sending u 0 only.")

    start_time = time.perf_counter()
    enabled_time: float | None = None
    last_valid_obs_time: float | None = None
    last_command_time = 0.0
    last_print_time = 0.0
    obs_count = 0
    sent_values: list[float] = []

    try:
        with serial.Serial(args.port, args.baud, timeout=0.01) as port:
            time.sleep(2.0)
            port.reset_input_buffer()
            send_u(port, 0.0)

            while True:
                now = time.perf_counter()

                if last_valid_obs_time is None:
                    if now - start_time > args.startup_timeout:
                        print("Safety stop: no valid obs received during startup.")
                        break
                elif now - last_valid_obs_time > args.timeout:
                    print("Safety stop: serial observation timeout.")
                    break

                raw = port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                obs = parse_obs(line)
                if obs is None:
                    continue

                obs_count += 1
                last_valid_obs_time = now

                phi = float(obs[3])
                phi_dot = float(obs[4])
                if abs(phi) > phi_limit:
                    print(f"Safety stop: phi={np.degrees(phi):.1f} deg exceeds limit.")
                    break
                if abs(phi_dot) > phi_dot_limit:
                    print(f"Safety stop: phi_dot={phi_dot:.2f} rad/s exceeds limit.")
                    break

                normalized_obs = vec_norm.normalize_obs(obs.reshape(1, -1))
                action, _ = model.predict(normalized_obs, deterministic=True)
                raw_u = float(np.asarray(action).reshape(-1)[0])
                scaled_u = raw_u * motor_scale
                clamped_u = float(np.clip(scaled_u, -action_limit, action_limit))

                if enabled_time is None:
                    if now - start_time < args.warmup:
                        sent_u = 0.0
                    else:
                        enabled_time = now
                        sent_u = clamped_u
                        print("Policy output enabled.")
                else:
                    if now - enabled_time >= args.seconds:
                        print("Limited live duration complete.")
                        break
                    sent_u = clamped_u

                if args.dry_run:
                    sent_u = 0.0

                # Do not spam serial faster than the firmware loop can consume.
                if now - last_command_time >= 0.02:
                    send_u(port, sent_u)
                    last_command_time = now
                    sent_values.append(sent_u)

                if now - last_print_time >= 0.1:
                    theta = float(np.arctan2(obs[1], obs[0]))
                    print(
                        f"theta={np.degrees(theta):+7.2f}deg "
                        f"theta_dot={obs[2]:+7.3f} "
                        f"phi={np.degrees(phi):+7.2f}deg "
                        f"phi_dot={obs[4]:+7.3f} "
                        f"raw_u={raw_u:+.3f} scaled_u={scaled_u:+.3f} sent_u={sent_u:+.3f}"
                    )
                    last_print_time = now

            send_u(port, 0.0)
            time.sleep(0.05)
            send_u(port, 0.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Sending stop if possible.")
        try:
            if "port" in locals() and port.is_open:
                send_u(port, 0.0)
        except Exception:
            pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        print("Close Arduino Serial Monitor and check the selected COM port.")
        return 1

    print(f"Done. Parsed {obs_count} observations.")
    if sent_values:
        sent = np.array(sent_values)
        print(
            "Sent action stats: "
            f"mean={sent.mean():+.4f}, min={sent.min():+.4f}, max={sent.max():+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
