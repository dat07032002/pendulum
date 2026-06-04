import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback,
    StopTrainingOnRewardThreshold,
)

from furuta_env import FurutaPendulumEnv

SAVE_DIR       = Path(__file__).with_name("models")
CLEAN_DIR      = SAVE_DIR / "clean"          # protected — never touched by train.py
N_ENVS         = 4
TOTAL_STEPS    = 3_000_000
EVAL_FREQ      = 10_000
CHECKPOINT_FREQ = 200_000

os.makedirs(CLEAN_DIR, exist_ok=True)


class SaveNormOnBest(BaseCallback):
    """Saves training VecNormalize stats every time EvalCallback finds a new best model."""
    def __init__(self, save_path: str, vec_env: VecNormalize):
        super().__init__()
        self._save_path = save_path
        self._vec_env   = vec_env

    def _on_step(self) -> bool:
        self._vec_env.save(self._save_path)
        return True


# --- training envs (no DR — clean replication of the original 799-reward run) ---
vec_env = make_vec_env(lambda: FurutaPendulumEnv(domain_rand=False), n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# --- eval env ---
eval_env = make_vec_env(lambda: FurutaPendulumEnv(domain_rand=False), n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                        clip_obs=10.0, training=False)

# --- model (identical hyperparams to the original run) ---
model = PPO(
    "MlpPolicy",
    vec_env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    learning_rate=3e-4,
    clip_range=0.2,
    verbose=1,
    device="cpu",
)

# --- callbacks ---
save_norm_cb = SaveNormOnBest(str(CLEAN_DIR / "vec_normalize_best.pkl"), vec_env)

stop_on_threshold = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)

eval_cb = EvalCallback(
    eval_env,
    callback_on_new_best=save_norm_cb,
    callback_after_eval=stop_on_threshold,
    best_model_save_path=str(CLEAN_DIR),
    log_path=str(CLEAN_DIR),
    eval_freq=max(EVAL_FREQ // N_ENVS, 1),
    n_eval_episodes=5,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
    save_path=str(CLEAN_DIR),
    name_prefix="ppo_furuta_clean",
)

# --- train ---
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

model.save(str(CLEAN_DIR / "ppo_furuta_clean_final"))
vec_env.save(str(CLEAN_DIR / "vec_normalize_final.pkl"))

# --- safe backup: timestamped copy that train.py can never touch ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = CLEAN_DIR / f"backup_{ts}"
backup_dir.mkdir()
for src in [
    CLEAN_DIR / "best_model.zip",
    CLEAN_DIR / "vec_normalize_best.pkl",
]:
    if src.exists():
        shutil.copy2(src, backup_dir / src.name)

print(f"\nDone. Clean best model -> {CLEAN_DIR}/best_model.zip")
print(f"Timestamped backup    -> {backup_dir}/")
print(f"\nWhen ready to run curriculum DR, train.py reads from models/clean/ automatically.")
