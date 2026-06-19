"""
Train a SAC BALANCE policy in the matched MuJoCo sim (sim_balance_env), for
sim-to-real deployment on the Nidec hardware.

The sim runs far faster than real time with full domain randomization, so this
can do millions of balance steps in minutes -- no cable pops, no recentering.
Deploy the result on hardware with:

  python run_policy.py --model-dir runs/sac_sim/<dir> --best --lift-to-catch --action-limit 1.0

The saved vec_normalize.pkl carries the obs normalization to hardware (obs
convention is identical: [cos th, sin th, th_dot, phi, phi_dot], th=0 upright).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim_balance_env import FurutaBalanceSimEnv

PROJECT_DIR = Path(__file__).parent


class SimEvalCheckpoint(BaseCallback):
    """Periodically eval (deterministic, DR off) and save latest + best."""

    def __init__(self, run_dir: Path, vec_env: VecNormalize, eval_every: int,
                 action_limit: float, episode_seconds: float, velocity_control: bool = True,
                 start_vel: float = 3.0, start_angle_deg: float = 10.0,
                 fall_threshold_deg: float = 45.0,
                 curriculum: bool = False, curriculum_steps: int = 0):
        super().__init__()
        self._run_dir = run_dir
        self._vec_env = vec_env
        self._eval_every = eval_every
        self._action_limit = action_limit
        self._episode_seconds = episode_seconds
        self._velocity_control = velocity_control
        self._start_vel = start_vel
        self._start_angle_deg = start_angle_deg
        self._fall_threshold_deg = fall_threshold_deg
        self._curriculum = curriculum
        self._curriculum_steps = curriculum_steps
        self._best_hold = -1.0
        self._last_eval = 0

    def _save(self, stem: str) -> None:
        self.model.save(str(self._run_dir / f"{stem}_model"))
        self._vec_env.save(str(self._run_dir / (f"{stem}_vec_normalize.pkl"
                                                if stem == "best" else "vec_normalize.pkl")))

    def _evaluate(self) -> tuple[float, float]:
        """Run a few deterministic episodes on a fresh DR-off env; return
        (mean reward, mean longest-hold seconds)."""
        env = FurutaBalanceSimEnv(domain_rand=False, action_limit=self._action_limit,
                                  episode_seconds=self._episode_seconds,
                                  velocity_control=self._velocity_control,
                                  start_vel=self._start_vel,
                                  start_angle_deg=self._start_angle_deg,
                                  fall_threshold_deg=self._fall_threshold_deg)
        rewards, holds = [], []
        for ep in range(5):
            obs, _ = env.reset(seed=1000 + ep)
            total, streak, best = 0.0, 0, 0
            done = False
            while not done:
                nobs = self._vec_env.normalize_obs(obs.reshape(1, -1))
                action, _ = self.model.predict(nobs, deterministic=True)
                obs, rew, term, trunc, _ = env.step(action.reshape(-1))
                total += rew
                theta = float(np.arctan2(obs[1], obs[0]))
                if abs(theta) < np.deg2rad(10.0) and abs(obs[2]) < 3.0:
                    streak += 1; best = max(best, streak)
                else:
                    streak = 0
                done = term or trunc
            rewards.append(total); holds.append(best * env.dt)
        env.close()
        return float(np.mean(rewards)), float(np.mean(holds))

    def _on_step(self) -> bool:
        if self._curriculum and self._curriculum_steps > 0:
            self.training_env.env_method("set_progress",
                                         min(1.0, self.num_timesteps / self._curriculum_steps))
        if self.num_timesteps - self._last_eval >= self._eval_every:
            self._last_eval = self.num_timesteps
            mean_rew, mean_hold = self._evaluate()
            self._save("latest")
            prog = min(1.0, self.num_timesteps / self._curriculum_steps) if self._curriculum and self._curriculum_steps else 1.0
            diff_tag = f"  diff={prog*100:.0f}%" if self._curriculum else ""
            tag = ""
            if mean_hold > self._best_hold:
                self._best_hold = mean_hold
                self._save("best")
                tag = "  <-- new best"
            print(f"[{self.num_timesteps:>8} steps] eval mean_reward={mean_rew:+8.1f}  "
                  f"mean_hold={mean_hold:.2f}s{diff_tag}{tag}", flush=True)
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a SAC balance policy in the matched sim.")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel sim envs for faster data")
    parser.add_argument("--no-domain-rand", action="store_true")
    parser.add_argument("--torque", action="store_true",
                        help="Torque control instead of Nidec speed control (diagnostic / alt motor)")
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--resume", default=None, help="Run dir to resume (loads latest)")
    parser.add_argument("--start-vel", type=float, default=15.0,
                        help="Max |theta_dot| at episode start [rad/s] — 15 matches hardware arrival speed")
    parser.add_argument("--start-angle", type=float, default=20.0,
                        help="Max |theta| at episode start [deg] — matches --switch-in-deg on hardware")
    parser.add_argument("--fall-threshold", type=float, default=45.0,
                        help="Terminate episode when |theta| exceeds this [deg] (default 45)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Ramp difficulty (arrival speed/angle + DR) easy->hard over training")
    parser.add_argument("--curriculum-frac", type=float, default=0.5,
                        help="Ramp to full difficulty over this fraction of total steps (default 0.5)")
    args = parser.parse_args()

    domain_rand = not args.no_domain_rand
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = PROJECT_DIR / run_dir
    else:
        run_dir = PROJECT_DIR / "runs" / "sac_sim" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sim SAC run dir: {run_dir}  (domain_rand={domain_rand}, n_envs={args.n_envs}, "
          f"start_vel={args.start_vel}, start_angle={args.start_angle})")

    velocity_control = not args.torque
    def make_env():
        return FurutaBalanceSimEnv(domain_rand=domain_rand, action_limit=args.action_limit,
                                   episode_seconds=args.episode_seconds,
                                   velocity_control=velocity_control,
                                   start_vel=args.start_vel,
                                   start_angle_deg=args.start_angle,
                                   fall_threshold_deg=args.fall_threshold,
                                   curriculum=args.curriculum)

    vec_env = DummyVecEnv([make_env for _ in range(args.n_envs)])

    if args.resume and (run_dir / "vec_normalize.pkl").exists():
        vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    sac_kwargs = dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=64,
        gradient_steps=64,
        ent_coef="auto",
        use_sde=True,
        sde_sample_freq=8,
    )

    if args.resume and (run_dir / "latest_model.zip").exists():
        model = SAC.load(str(run_dir / "latest_model"), env=vec_env, device="auto")
        print(f"Resumed from {model.num_timesteps} steps")
    else:
        model = SAC("MlpPolicy", vec_env, verbose=0, device="auto", **sac_kwargs)

    curriculum_steps = int(args.curriculum_frac * args.total_steps) if args.curriculum else 0
    cb = SimEvalCheckpoint(run_dir, vec_env, args.eval_every, args.action_limit,
                           args.episode_seconds, velocity_control=velocity_control,
                           start_vel=args.start_vel, start_angle_deg=args.start_angle,
                           fall_threshold_deg=args.fall_threshold,
                           curriculum=args.curriculum, curriculum_steps=curriculum_steps)
    try:
        model.learn(total_timesteps=args.total_steps, reset_num_timesteps=not args.resume,
                    progress_bar=True, callback=cb)
    finally:
        model.save(str(run_dir / "latest_model"))
        vec_env.save(str(run_dir / "vec_normalize.pkl"))
        print(f"Saved final model + normalization to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
