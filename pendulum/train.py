import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback, CallbackList,
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold,
)

from furuta_env import FurutaPendulumEnv

SAVE_DIR        = Path(__file__).with_name("models")
CLEAN_DIR       = SAVE_DIR / "clean"   # read-only source; train_clean.py writes here
N_ENVS          = 4
TOTAL_STEPS     = 4_000_000
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


class CurriculumCallback(BaseCallback):
    """Linearly ramps DR progress from 0 → 1 over curriculum_steps timesteps."""
    def __init__(self, schedule: dict, curriculum_steps: int):
        super().__init__()
        self._schedule         = schedule
        self._curriculum_steps = curriculum_steps

    def _on_step(self) -> bool:
        self._schedule["progress"] = min(1.0, self.num_timesteps / self._curriculum_steps)
        return True


# --- training envs ---
schedule = {"progress": 0.0}
vec_env = make_vec_env(lambda: FurutaPendulumEnv(dr_schedule=schedule), n_envs=N_ENVS)
# warm-start: load clean model's normalization stats (written by train_clean.py)
vec_env = VecNormalize.load(str(CLEAN_DIR / "vec_normalize_best.pkl"), vec_env)
vec_env.training    = True
vec_env.norm_reward = True

# --- eval env (no DR — clean comparison) ---
eval_env = make_vec_env(lambda: FurutaPendulumEnv(domain_rand=False), n_envs=1)
eval_env = VecNormalize.load(str(CLEAN_DIR / "vec_normalize_best.pkl"), eval_env)
eval_env.training   = False
eval_env.norm_reward = False

# --- model: warm-start from clean best model (train_clean.py output) ---
# Higher ent_coef (0.05 vs 0.01) prevents entropy collapse under the harder DR task.
# Lower learning_rate (1e-4 vs 3e-4) preserves swing-up knowledge while adapting.
model = PPO.load(
    str(CLEAN_DIR / "best_model"),
    env=vec_env,
    device="cpu",
    ent_coef=0.005,
    learning_rate=1e-4,
)

# --- callbacks ---
# SaveNormOnBest fires via callback_on_new_best — stats are always in sync with best model
save_norm_cb = SaveNormOnBest(str(SAVE_DIR / "vec_normalize_best.pkl"), vec_env)

# Stop early if reward exceeds 700 (strong result — no need to keep going)
stop_on_threshold = StopTrainingOnRewardThreshold(reward_threshold=999, verbose=1)

# Stop if no new best for 50 consecutive evals (500k steps) — plateau detected
# min_evals=50 prevents stopping in the first 500k steps while policy is still adapting
stop_on_plateau = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=100, min_evals=150, verbose=1
)

eval_cb = EvalCallback(
    eval_env,
    callback_on_new_best=CallbackList([save_norm_cb, stop_on_threshold]),
    callback_after_eval=stop_on_plateau,
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

# curriculum reaches full DR at 2M steps, stays there for the final 1M
curriculum_cb = CurriculumCallback(schedule, curriculum_steps=3_000_000)

# --- train ---
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb, curriculum_cb],
    progress_bar=True,
)

model.save(str(SAVE_DIR / "ppo_furuta_final"))
vec_env.save(str(SAVE_DIR / "vec_normalize.pkl"))
print(f"\nDone. Saved to {SAVE_DIR}/")
