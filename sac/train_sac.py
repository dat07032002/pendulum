import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from furuta_env import FurutaPendulumEnv

PROJECT_DIR = Path(__file__).parent
RUNS_DIR = PROJECT_DIR / "runs" / "sac_no_dr"
RUN_ID = os.environ.get("FURUTA_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = RUNS_DIR / RUN_ID

N_ENVS = 4
TOTAL_STEPS = 1_000_000          # SAC is off-policy: needs far fewer steps than PPO's 3M
EPISODE_SECONDS = 30.0
EVAL_FREQ = 10_000
EVAL_EPISODES = 20
CHECKPOINT_FREQ = 100_000
REWARD_THRESHOLD = float(os.environ.get("FURUTA_REWARD_THRESHOLD", "2550"))
FIXED_MOTOR_DEADBAND = float(os.environ.get("FURUTA_FIXED_MOTOR_DEADBAND", "0.0"))
ACTION_LIMIT = float(os.environ.get("FURUTA_ACTION_LIMIT", "1.0"))

SAC_KWARGS = dict(
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=10_000,      # warm up the buffer with random actions first
    batch_size=256,
    tau=0.005,                   # soft target-network update rate
    gamma=0.99,
    train_freq=1,
    gradient_steps=-1,           # one gradient step per env transition collected
    ent_coef="auto",             # automatic entropy temperature tuning
)

RUN_DIR.mkdir(parents=True, exist_ok=True)
if (RUN_DIR / "run_config.json").exists():
    raise FileExistsError(f"Run directory already contains a training run: {RUN_DIR}")
shutil.copy2(PROJECT_DIR / "furuta_pendulum.xml", RUN_DIR / "furuta_pendulum.xml")
(RUN_DIR / "run_config.json").write_text(
    json.dumps(
        {
            "run_id": RUN_ID,
            "algorithm": "SAC",
            "domain_randomization": False,
            "n_envs": N_ENVS,
            "total_steps": TOTAL_STEPS,
            "episode_seconds": EPISODE_SECONDS,
            "eval_freq": EVAL_FREQ,
            "eval_episodes": EVAL_EPISODES,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "reward_threshold": REWARD_THRESHOLD,
            "fixed_motor_deadband": FIXED_MOTOR_DEADBAND,
            "action_limit": ACTION_LIMIT,
            "sac": SAC_KWARGS,
        },
        indent=2,
    )
    + "\n",
    encoding="ascii",
)


class SaveNormOnBest(BaseCallback):
    """Save training VecNormalize stats whenever evaluation finds a new best model."""

    def __init__(self, save_path: str, vec_env: VecNormalize):
        super().__init__()
        self._save_path = save_path
        self._vec_env = vec_env

    def _on_step(self) -> bool:
        self._vec_env.save(self._save_path)
        return True


# Training and evaluation both use the nominal model with no domain randomization.
# norm_reward must stay False for off-policy algorithms: replay-buffer transitions
# would otherwise carry rewards scaled by stale normalization statistics.
vec_env = make_vec_env(
    lambda: FurutaPendulumEnv(
        domain_rand=False,
        episode_seconds=EPISODE_SECONDS,
        fixed_motor_deadband=FIXED_MOTOR_DEADBAND,
        action_limit=ACTION_LIMIT,
    ),
    n_envs=N_ENVS,
)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

eval_env = make_vec_env(
    lambda: FurutaPendulumEnv(
        domain_rand=False,
        episode_seconds=EPISODE_SECONDS,
        fixed_motor_deadband=FIXED_MOTOR_DEADBAND,
        action_limit=ACTION_LIMIT,
    ),
    n_envs=1,
)
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    training=False,
)

model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    device="cpu",
    **SAC_KWARGS,
)

save_norm_cb = SaveNormOnBest(str(RUN_DIR / "vec_normalize_best.pkl"), vec_env)
stop_on_threshold = StopTrainingOnRewardThreshold(
    reward_threshold=REWARD_THRESHOLD,
    verbose=1,
)
stop_on_plateau = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=100,
    min_evals=150,
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    callback_on_new_best=CallbackList([save_norm_cb, stop_on_threshold]),
    callback_after_eval=stop_on_plateau,
    best_model_save_path=str(RUN_DIR),
    log_path=str(RUN_DIR),
    eval_freq=max(EVAL_FREQ // N_ENVS, 1),
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
    save_path=str(RUN_DIR),
    name_prefix="sac_furuta_clean",
)

print(f"SAC no-DR run directory: {RUN_DIR}")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

model.save(str(RUN_DIR / "sac_furuta_clean_final"))
vec_env.save(str(RUN_DIR / "vec_normalize_final.pkl"))

print(f"\nDone. SAC run  -> {RUN_DIR}")
print(f"Best model     -> {RUN_DIR}/best_model.zip")
