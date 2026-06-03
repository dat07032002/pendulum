import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from furuta_env import FurutaPendulumEnv

SAVE_DIR        = Path(__file__).with_name("models")
N_ENVS          = 4
TOTAL_STEPS     = 3_000_000
EVAL_FREQ       = 10_000
CHECKPOINT_FREQ = 200_000

os.makedirs(SAVE_DIR, exist_ok=True)


class SaveNormOnBest(BaseCallback):
    """Saves training VecNormalize stats every time EvalCallback finds a new best model."""
    def __init__(self, save_path: str, vec_env: VecNormalize):
        super().__init__()
        self._save_path = save_path
        self._vec_env   = vec_env

    def _on_step(self) -> bool:
        self._vec_env.save(self._save_path)
        return True


# --- training envs ---
vec_env = make_vec_env(FurutaPendulumEnv, n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# --- eval env ---
eval_env = make_vec_env(FurutaPendulumEnv, n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                        clip_obs=10.0, training=False)

# --- model ---
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
    tensorboard_log=None,
)

# --- callbacks ---
# SaveNormOnBest fires via callback_on_new_best — stats are always in sync with best model
save_norm_cb = SaveNormOnBest(str(SAVE_DIR / "vec_normalize_best.pkl"), vec_env)

eval_cb = EvalCallback(
    eval_env,
    callback_on_new_best=save_norm_cb,
    best_model_save_path=str(SAVE_DIR),
    log_path=str(SAVE_DIR),
    eval_freq=max(EVAL_FREQ // N_ENVS, 1),
    n_eval_episodes=5,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
    save_path=str(SAVE_DIR),
    name_prefix="ppo_furuta",
)

# --- train ---
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb],
    progress_bar=True,
)

model.save(str(SAVE_DIR / "ppo_furuta_final"))
vec_env.save(str(SAVE_DIR / "vec_normalize.pkl"))
print(f"\nDone. Saved to {SAVE_DIR}/")
