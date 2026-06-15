"""
Train SAC directly on the real Furuta pendulum over the ESP32 serial bridge.

Key design points for live-hardware training:
  - train_freq=(1, "episode"): gradient updates run between episodes while the
    motor is stopped, so the 50 Hz control loop is never blocked by learning.
  - The replay buffer is saved with every checkpoint, so a crash or Ctrl+C
    loses no robot experience: resume with --resume <run_dir>.
  - Random warm-up actions and all policy actions are clamped by the env's
    action_limit; phi / phi_dot safety stops end the episode with the motor off.

Start conservative:
  python train_sac_hw.py --port COM5 --action-limit 0.6 --episode-seconds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from furuta_hw_env import FurutaHardwareEnv

PROJECT_DIR = Path(__file__).parent


class HardwareCheckpoint(BaseCallback):
    """Save model + replay buffer + VecNormalize stats every n episodes."""

    def __init__(self, run_dir: Path, vec_env: VecNormalize, every_episodes: int = 5):
        super().__init__()
        self._run_dir = run_dir
        self._vec_env = vec_env
        self._every = every_episodes
        self._episodes = 0

    def _on_step(self) -> bool:
        for done in self.locals.get("dones", []):
            if done:
                self._episodes += 1
                ep_rew = None
                infos = self.locals.get("infos", [])
                if infos and "episode" in infos[0]:
                    ep_rew = infos[0]["episode"]["r"]
                print(f"--- episode {self._episodes} done"
                      + (f", reward={ep_rew:.1f}" if ep_rew is not None else ""))
                if self._episodes % self._every == 0:
                    self.save_all()
        return True

    def save_all(self) -> None:
        self.model.save(str(self._run_dir / "latest_model"))
        self.model.save_replay_buffer(str(self._run_dir / "replay_buffer"))
        self._vec_env.save(str(self._run_dir / "vec_normalize.pkl"))
        print(f"Checkpoint saved to {self._run_dir} (step {self.num_timesteps})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live SAC training on the real Furuta pendulum.")
    parser.add_argument("--port", default="COM5", help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--control-dt", type=float, default=0.02, help="Control period in seconds (50 Hz default)")
    parser.add_argument("--episode-seconds", type=float, default=15.0)
    parser.add_argument("--action-limit", type=float, default=1.0, help="Clamp actions to +/- this value")
    parser.add_argument("--phi-limit-deg", type=float, default=90.0, help="Safety stop if |phi| exceeds this")
    parser.add_argument("--total-steps", type=int, default=100_000, help="~33 min of robot time at 50 Hz")
    parser.add_argument("--learning-starts", type=int, default=1_500, help="Random warm-up steps before learning")
    parser.add_argument("--checkpoint-episodes", type=int, default=5)
    parser.add_argument("--manual-recenter", action="store_true",
                        help="User recenters the arm by hand between episodes (no automatic motor pulses)")
    parser.add_argument("--resume", default=None, help="Run directory to resume from")
    parser.add_argument("--warm-start", default=None,
                        help="Run directory to load policy weights from, with a FRESH replay buffer "
                             "(use after changing motor dynamics, e.g. firmware deadband compensation)")
    args = parser.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = PROJECT_DIR / run_dir
        if not (run_dir / "latest_model.zip").exists():
            raise FileNotFoundError(f"No latest_model.zip in {run_dir}")
    else:
        run_dir = PROJECT_DIR / "runs" / "sac_hw" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return Monitor(FurutaHardwareEnv(
            port=args.port,
            baud=args.baud,
            control_dt=args.control_dt,
            episode_seconds=args.episode_seconds,
            action_limit=args.action_limit,
            phi_limit_deg=args.phi_limit_deg,
            recenter=not args.manual_recenter,
        ))

    vec_env = DummyVecEnv([make_env])

    sac_kwargs = dict(
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=args.learning_starts,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),   # learn between episodes, never inside the control loop
        gradient_steps=-1,           # one gradient step per transition collected
        ent_coef="auto",
        use_sde=True,                # smooth state-dependent exploration (kinder to real motors)
        sde_sample_freq=8,
    )

    if args.resume:
        vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
        vec_env.training = True
        vec_env.norm_reward = False
        model = SAC.load(str(run_dir / "latest_model"), env=vec_env, device="cpu")
        model.load_replay_buffer(str(run_dir / "replay_buffer"))
        print(f"Resumed from {run_dir}: {model.num_timesteps} steps, "
              f"{model.replay_buffer.size()} transitions in buffer")
    elif args.warm_start:
        warm_dir = Path(args.warm_start)
        if not warm_dir.is_absolute():
            warm_dir = PROJECT_DIR / warm_dir
        # Observation distribution is unchanged by a motor remap: reuse stats.
        vec_env = VecNormalize.load(str(warm_dir / "vec_normalize.pkl"), vec_env)
        vec_env.training = True
        vec_env.norm_reward = False
        model = SAC.load(str(warm_dir / "latest_model"), env=vec_env, device="cpu")
        # Fresh buffer (not loaded) and a short re-exploration phase under the
        # new dynamics before gradient updates resume.
        model.learning_starts = args.learning_starts
        print(f"Warm start from {warm_dir}: policy weights loaded, replay buffer FRESH")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = SAC("MlpPolicy", vec_env, verbose=1, device="cpu", **sac_kwargs)

    if not args.resume:
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "algorithm": "SAC",
                    "hardware": True,
                    "port": args.port,
                    "control_dt": args.control_dt,
                    "episode_seconds": args.episode_seconds,
                    "action_limit": args.action_limit,
                    "phi_limit_deg": args.phi_limit_deg,
                    "manual_recenter": args.manual_recenter,
                    "warm_start": args.warm_start,
                    "total_steps": args.total_steps,
                    "sac": {k: str(v) for k, v in sac_kwargs.items()},
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )

    ckpt_cb = HardwareCheckpoint(run_dir, vec_env, every_episodes=args.checkpoint_episodes)

    print(f"Hardware SAC run directory: {run_dir}")
    print(f"Control rate {1.0 / args.control_dt:.0f} Hz, episodes {args.episode_seconds:.0f}s, "
          f"action clamp +/-{args.action_limit:.2f}")
    print("Ctrl+C stops the motor and saves a checkpoint.")

    # Start gate: the serial port is open and the encoder is zeroed at the
    # arm's current (centered) pose, but no episode may begin until the user
    # confirms they are present and watching the log.
    print(">>> Press Enter in the training window to begin training...", flush=True)
    try:
        input()
    except EOFError:
        import time as _time
        print(">>> No interactive stdin; starting in 10 seconds (Ctrl+C to abort)...", flush=True)
        _time.sleep(10.0)
    print(">>> Training started.", flush=True)

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=ckpt_cb,
            reset_num_timesteps=not args.resume,
            progress_bar=sys.stdout.isatty(),  # no progress bar when logging to a file
        )
    except KeyboardInterrupt:
        print("\nInterrupted: saving checkpoint and stopping motor.")
    finally:
        ckpt_cb.model = model  # ensure save works even if interrupted before first callback step
        ckpt_cb.save_all()
        vec_env.close()  # closes the env -> sends u 0 and releases the port

    print(f"\nDone. Resume any time with:\n  python train_sac_hw.py --resume {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
