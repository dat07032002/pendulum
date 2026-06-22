"""
Train a SAC balance policy in the matched MuJoCo simulation.

The policy is trained for the energy-gated catch and balance region, then
deployed through run_swingup_balance.py with matching start-angle and
start-velocity bounds.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from sim_balance_env import FurutaBalanceSimEnv

PROJECT_DIR = Path(__file__).parent


class SimEvalCheckpoint(BaseCallback):
    """Periodically evaluate without DR and save latest and best policies."""

    def __init__(
        self,
        run_dir: Path,
        vec_env: VecNormalize,
        eval_every: int,
        action_limit: float,
        episode_seconds: float,
        velocity_control: bool = True,
        start_vel: float = 3.0,
        start_angle_deg: float = 10.0,
        fall_threshold_deg: float = 45.0,
        curriculum: bool = False,
        curriculum_steps: int = 0,
        corridor_reset: bool = True,
        switch_dE: float = 0.08,
    ):
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
        self._corridor_reset = corridor_reset
        self._switch_dE = switch_dE
        self._best_hold = -1.0
        self._last_eval = 0

    def _save(self, stem: str) -> None:
        self.model.save(str(self._run_dir / f"{stem}_model"))
        norm_name = f"{stem}_vec_normalize.pkl" if stem == "best" else "vec_normalize.pkl"
        self._vec_env.save(str(self._run_dir / norm_name))

    def _evaluate(self) -> tuple[float, float]:
        """Return mean reward and mean longest upright-hold duration."""
        env = FurutaBalanceSimEnv(
            domain_rand=False,
            action_limit=self._action_limit,
            episode_seconds=self._episode_seconds,
            velocity_control=self._velocity_control,
            start_vel=self._start_vel,
            start_angle_deg=self._start_angle_deg,
            fall_threshold_deg=self._fall_threshold_deg,
            corridor_reset=self._corridor_reset,
            switch_dE=self._switch_dE,
        )
        rewards, holds = [], []
        for episode in range(5):
            obs, _ = env.reset(seed=1000 + episode)
            total, streak, best = 0.0, 0, 0
            done = False
            while not done:
                normalized_obs = self._vec_env.normalize_obs(obs.reshape(1, -1))
                action, _ = self.model.predict(normalized_obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action.reshape(-1))
                total += reward
                theta = float(np.arctan2(obs[1], obs[0]))
                if abs(theta) < np.deg2rad(10.0) and abs(obs[2]) < 3.0:
                    streak += 1
                    best = max(best, streak)
                else:
                    streak = 0
                done = terminated or truncated
            rewards.append(total)
            holds.append(best * env.dt)
        env.close()
        return float(np.mean(rewards)), float(np.mean(holds))

    def _on_step(self) -> bool:
        if self._curriculum and self._curriculum_steps > 0:
            progress = min(1.0, self.num_timesteps / self._curriculum_steps)
            self.training_env.env_method("set_progress", progress)

        if self.num_timesteps - self._last_eval < self._eval_every:
            return True

        self._last_eval = self.num_timesteps
        mean_reward, mean_hold = self._evaluate()
        self._save("latest")
        progress = (
            min(1.0, self.num_timesteps / self._curriculum_steps)
            if self._curriculum and self._curriculum_steps
            else 1.0
        )
        difficulty = f"  diff={progress * 100:.0f}%" if self._curriculum else ""
        tag = ""
        if mean_hold > self._best_hold:
            self._best_hold = mean_hold
            self._save("best")
            tag = "  <-- new best"
        print(
            f"[{self.num_timesteps:>8} steps] eval mean_reward={mean_reward:+8.1f} "
            f" mean_hold={mean_hold:.2f}s{difficulty}{tag}",
            flush=True,
        )
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a SAC balance policy in the matched simulation.")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel simulation environments")
    parser.add_argument("--no-domain-rand", action="store_true")
    parser.add_argument("--torque", action="store_true", help="Use the voltage/torque motor model")
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--resume", default=None, help="Resume latest model in an existing run directory")
    parser.add_argument(
        "--warm-start",
        default=None,
        help="Load policy weights into a fresh run with a fresh replay buffer and curriculum",
    )
    parser.add_argument("--warm-start-best", action="store_true", help="Load best_model for warm start")
    parser.add_argument(
        "--start-vel",
        type=float,
        default=10.0,
        help="Maximum initial pendulum velocity; match the energy-gated hardware delivery",
    )
    parser.add_argument(
        "--start-angle",
        type=float,
        default=30.0,
        help="Maximum initial angle; match deployment switch-in-deg",
    )
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=55.0,
        help="Terminate when pendulum angle exceeds this value in degrees",
    )
    parser.add_argument(
        "--switch-dE",
        type=float,
        default=0.08,
        help="Energy-gate tolerance (fraction of E_max) defining the handoff corridor. "
             "Match run_swingup_balance.py --switch-dE.",
    )
    parser.add_argument(
        "--rectangle-reset",
        action="store_true",
        help="Sample starts from the legacy independent theta x theta_dot rectangle "
             "instead of the energy-gated corridor.",
    )
    parser.add_argument("--curriculum", action="store_true", help="Ramp arrival difficulty and DR")
    parser.add_argument(
        "--curriculum-frac",
        type=float,
        default=0.5,
        help="Fraction of training steps used to reach full curriculum difficulty",
    )
    # --- SAC stability knobs (defaults chosen to avoid the long-run collapse) ---
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="SAC learning rate (1e-4 is gentler/steadier than 3e-4).")
    parser.add_argument("--train-freq", type=int, default=64,
                        help="Env steps between update phases.")
    parser.add_argument("--gradient-steps", type=int, default=32,
                        help="Gradient updates per phase. Lower than train-freq (<1.0 ratio) is "
                             "less aggressive and more stable on long runs.")
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="Entropy coefficient: 'auto', 'auto_<init>', or a fixed float "
                             "(e.g. 0.05) to stop the temperature from blowing up.")
    parser.add_argument("--use-sde", action="store_true",
                        help="Enable gSDE exploration. Off by default — gSDE was a likely "
                             "contributor to the policy collapse.")
    args = parser.parse_args()

    domain_rand = not args.no_domain_rand
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = PROJECT_DIR / run_dir
    else:
        run_dir = PROJECT_DIR / "runs" / "sac_sim" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Sim SAC run dir: {run_dir} (domain_rand={domain_rand}, n_envs={args.n_envs}, "
        f"start_vel={args.start_vel}, start_angle={args.start_angle})"
    )

    velocity_control = not args.torque
    corridor_reset = not args.rectangle_reset

    def make_env():
        return FurutaBalanceSimEnv(
            domain_rand=domain_rand,
            action_limit=args.action_limit,
            episode_seconds=args.episode_seconds,
            velocity_control=velocity_control,
            start_vel=args.start_vel,
            start_angle_deg=args.start_angle,
            fall_threshold_deg=args.fall_threshold,
            curriculum=args.curriculum,
            corridor_reset=corridor_reset,
            switch_dE=args.switch_dE,
        )

    vec_env = DummyVecEnv([make_env for _ in range(args.n_envs)])

    warm_start_dir = None
    if args.warm_start:
        warm_start_dir = Path(args.warm_start)
        if not warm_start_dir.is_absolute():
            warm_start_dir = PROJECT_DIR / warm_start_dir

    if args.resume and (run_dir / "vec_normalize.pkl").exists():
        vec_env = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), vec_env)
        vec_env.training = True
    elif warm_start_dir is not None:
        norm_name = "best_vec_normalize.pkl" if args.warm_start_best else "vec_normalize.pkl"
        vec_env = VecNormalize.load(str(warm_start_dir / norm_name), vec_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    try:
        ent_coef = float(args.ent_coef)   # fixed coefficient
    except ValueError:
        ent_coef = args.ent_coef          # "auto" / "auto_<init>"

    sac_kwargs = {
        "learning_rate": args.learning_rate,
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "batch_size": 512,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "ent_coef": ent_coef,
        "use_sde": args.use_sde,
        "sde_sample_freq": 8,
    }
    print(f"SAC: lr={args.learning_rate} train_freq={args.train_freq} "
          f"gradient_steps={args.gradient_steps} ent_coef={ent_coef} use_sde={args.use_sde}")

    if args.resume and (run_dir / "latest_model.zip").exists():
        model = SAC.load(str(run_dir / "latest_model"), env=vec_env, device="auto")
        print(f"Resumed from {model.num_timesteps} steps")
    elif warm_start_dir is not None:
        model_name = "best_model" if args.warm_start_best else "latest_model"
        model = SAC("MlpPolicy", vec_env, verbose=0, device="auto", **sac_kwargs)
        model.set_parameters(str(warm_start_dir / model_name))
        print(f"Warm-started weights from {warm_start_dir / model_name} (fresh buffer + curriculum)")
    else:
        model = SAC("MlpPolicy", vec_env, verbose=0, device="auto", **sac_kwargs)

    curriculum_steps = int(args.curriculum_frac * args.total_steps) if args.curriculum else 0
    callback = SimEvalCheckpoint(
        run_dir,
        vec_env,
        args.eval_every,
        args.action_limit,
        args.episode_seconds,
        velocity_control=velocity_control,
        start_vel=args.start_vel,
        start_angle_deg=args.start_angle,
        fall_threshold_deg=args.fall_threshold,
        curriculum=args.curriculum,
        curriculum_steps=curriculum_steps,
        corridor_reset=corridor_reset,
        switch_dE=args.switch_dE,
    )
    try:
        model.learn(
            total_timesteps=args.total_steps,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
            callback=callback,
        )
    finally:
        model.save(str(run_dir / "latest_model"))
        vec_env.save(str(run_dir / "vec_normalize.pkl"))
        print(f"Saved final model + normalization to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
