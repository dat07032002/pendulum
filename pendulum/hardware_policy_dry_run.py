"""
Dry-run a trained PPO policy on real ESP32 observations.

This reads obs=[cos(theta),sin(theta),theta_dot,phi,phi_dot] from the ESP32,
normalizes the observation with the saved VecNormalize statistics, predicts the
PPO action, prints it, and always sends "u 0" to the ESP32.

No policy action is applied to the motor in this script.
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
except ImportError:  # pragma: no cover - user environment helper
    print("Missing dependency: pyserial")
    print("Install it with: python -m pip install pyserial")
    sys.exit(1)

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:  # pragma: no cover - user environment helper
    print("Missing dependency: stable-baselines3/gymnasium")
    print("Install your training dependencies before running policy dry-run.")
    sys.exit(1)


OBS_RE = re.compile(r"obs=\[([^\]]+)\]")
DEFAULT_MODEL_DIR = Path("trained_policies") / "dr_hardware_masses_20260609"


class PolicySpaceEnv(gym.Env):
    """Minimal env used only so VecNormalize can load saved statistics."""

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
        obs = np.array([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None

    return obs


def send_stop(port: serial.Serial) -> None:
    port.write(b"u 0\n")
    port.flush()


def load_policy(model_dir: Path) -> tuple[PPO, VecNormalize]:
    model_path = model_dir / "best_model.zip"
    norm_path = model_dir / "vec_normalize_best.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats: {norm_path}")

    model = PPO.load(model_path)

    dummy_env = DummyVecEnv([PolicySpaceEnv])
    vec_norm = VecNormalize.load(norm_path, dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict policy actions from real ESP32 obs, but send u 0.")
    parser.add_argument("--port", default="COM5", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--seconds", type=float, default=20.0, help="How long to run")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Folder with best_model.zip and vec_normalize_best.pkl")
    parser.add_argument("--stop-hz", type=float, default=10.0, help="How often to send u 0")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent / model_dir

    print(f"Loading policy from {model_dir}")
    model, vec_norm = load_policy(model_dir)

    print(f"Opening {args.port} at {args.baud} baud")
    print("DRY RUN ONLY: predicted policy actions are printed, but ESP32 receives u 0.")
    print("Close Arduino Serial Monitor before running this.")

    start_time = time.perf_counter()
    last_stop_time = 0.0
    last_print_time = 0.0
    obs_count = 0
    action_values: list[float] = []

    try:
        with serial.Serial(args.port, args.baud, timeout=0.02) as port:
            time.sleep(2.0)
            port.reset_input_buffer()
            send_stop(port)

            while time.perf_counter() - start_time < args.seconds:
                now = time.perf_counter()

                if now - last_stop_time >= 1.0 / args.stop_hz:
                    send_stop(port)
                    last_stop_time = now

                raw = port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                obs = parse_obs(line)
                if obs is None:
                    continue

                obs_count += 1
                normalized_obs = vec_norm.normalize_obs(obs.reshape(1, -1))
                action, _ = model.predict(normalized_obs, deterministic=True)
                action_u = float(np.asarray(action).reshape(-1)[0])
                action_values.append(action_u)

                if now - last_print_time >= 0.2:
                    theta = float(np.arctan2(obs[1], obs[0]))
                    phi = float(obs[3])
                    rate = obs_count / max(now - start_time, 1e-9)
                    print(
                        f"rate={rate:5.1f}Hz "
                        f"theta={np.degrees(theta):+7.2f}deg "
                        f"theta_dot={obs[2]:+7.3f} "
                        f"phi={np.degrees(phi):+7.2f}deg "
                        f"phi_dot={obs[4]:+7.3f} "
                        f"policy_u={action_u:+.4f} sent_u=0"
                    )
                    last_print_time = now

            send_stop(port)

    except KeyboardInterrupt:
        print("\nInterrupted. Sending stop if possible.")
        try:
            if "port" in locals() and port.is_open:
                send_stop(port)
        except Exception:
            pass
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        print("Check that Arduino Serial Monitor is closed and the ESP32 is on the selected COM port.")
        return 1

    print(f"Done. Parsed {obs_count} observations.")
    if action_values:
        actions = np.array(action_values)
        print(
            "Predicted action stats: "
            f"mean={actions.mean():+.4f}, "
            f"std={actions.std():.4f}, "
            f"min={actions.min():+.4f}, "
            f"max={actions.max():+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
