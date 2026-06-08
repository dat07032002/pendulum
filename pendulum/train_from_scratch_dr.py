"""
Train PPO from scratch with curriculum domain randomisation.

Unlike train.py (warm-start + DR), this script starts with a fresh policy and
ramps DR from zero, avoiding the catastrophic forgetting observed in warm-start runs.
The curriculum ramp completes at 1.5M steps so the policy spends the second half
of training at full DR, giving it time to consolidate robustness.

Expected behaviour:
  - First ~50–100k steps: random policy, reward ~-1000
  - ~50–200k: swing-up emerges
  - ~500k–1.5M: balance consolidates under growing DR
  - 1.5M–3M: full DR, policy stabilises or improves further

Outputs go to models/scratch_dr/ (never touches models/clean/).
"""
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

SAVE_DIR        = Path(__file__).with_name("models") / "scratch_dr"
N_ENVS          = 4
TOTAL_STEPS     = 3_000_000
CURRICULUM_STEPS = 1_500_000   # full DR reached at 1.5M; train at full DR for remaining 1.5M
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


# --- training envs (fresh VecNormalize — no warm-start stats) ---
schedule = {"progress": 0.0}
vec_env = make_vec_env(lambda: FurutaPendulumEnv(dr_schedule=schedule), n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# --- eval env (no DR — clean comparison across all runs) ---
eval_env = make_vec_env(lambda: FurutaPendulumEnv(domain_rand=False), n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                        clip_obs=10.0, training=False)

# --- fresh model (identical hyperparams to clean training) ---
model = PPO(
    "MlpPolicy",
    vec_env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,        # same as clean — policy must discover swing-up, so exploration matters
    learning_rate=3e-4,   # same as clean — no warm-start to protect
    clip_range=0.2,
    verbose=1,
    device="cpu",
)

# --- callbacks ---
save_norm_cb = SaveNormOnBest(str(SAVE_DIR / "vec_normalize_best.pkl"), vec_env)

# Stop if reward ever reaches 900 (genuine success)
stop_on_threshold = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)

# Stop if no new best for 100 consecutive evals (1M steps) — plateau detected
# min_evals=100 prevents stopping before curriculum finishes ramping (1.5M / 10k = 150 evals)
stop_on_plateau = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=100, min_evals=100, verbose=1
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
    name_prefix="ppo_furuta_scratch_dr",
)

curriculum_cb = CurriculumCallback(schedule, curriculum_steps=CURRICULUM_STEPS)

# --- train ---
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb, curriculum_cb],
    progress_bar=True,
)

model.save(str(SAVE_DIR / "ppo_furuta_scratch_dr_final"))
vec_env.save(str(SAVE_DIR / "vec_normalize_final.pkl"))
print(f"\nDone. Best model -> {SAVE_DIR}/best_model.zip")
print(f"To evaluate: python eval.py {SAVE_DIR}/best_model")
