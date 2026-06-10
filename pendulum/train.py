import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
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
NO_DR_RUNS_DIR = PROJECT_DIR / "runs" / "no_dr"
RUNS_DIR = PROJECT_DIR / "runs" / "dr"
RUN_ID = os.environ.get("FURUTA_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
DR_PROFILE = os.environ.get("FURUTA_DR_PROFILE", "real_ready_stage2")
RUN_DIR = RUNS_DIR / RUN_ID

N_ENVS = int(os.environ.get("FURUTA_N_ENVS", "4"))
DEVICE  = os.environ.get("FURUTA_DEVICE", "auto")
TOTAL_STEPS = 3_000_000
CURRICULUM_STEPS = 1_500_000
EPISODE_SECONDS = float(os.environ.get("FURUTA_EPISODE_SECONDS", "30.0"))
EVAL_FREQ = 10_000
EVAL_EPISODES = 20
CHECKPOINT_FREQ = 200_000
REWARD_THRESHOLD = float(os.environ.get("FURUTA_REWARD_THRESHOLD", "3_550"))
ENT_COEF = float(os.environ.get("FURUTA_ENT_COEF", "0.01"))
ELBOW_KICK_DEG = float(os.environ.get("FURUTA_ELBOW_KICK_DEG", "0.0"))
ELBOW_KICK_COUNT = int(os.environ.get("FURUTA_ELBOW_KICK_COUNT", "1"))
FALL_THRESHOLD_DEG = float(os.environ.get("FURUTA_FALL_THRESHOLD_DEG", "20.0"))
VALID_DR_PROFILES = {
    "all", "mass", "shoulder_damping", "elbow_damping",
    "motor", "delay", "sensor", "sensor_pos", "sensor_vel",
    "sensor_filter", "sensor_elbow_pos_small", "sensor_filter_narrow",
    "mass_motor_delay", "real_ready", "real_ready_stage2", "none",
}
if DR_PROFILE not in VALID_DR_PROFILES:
    raise ValueError(f"Unknown FURUTA_DR_PROFILE={DR_PROFILE!r}; expected one of {sorted(VALID_DR_PROFILES)}")


def latest_no_dr_run() -> Path:
    runs = sorted(p for p in NO_DR_RUNS_DIR.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No no-DR runs found in {NO_DR_RUNS_DIR}")
    return runs[-1]


WARM_START_DIR = Path(os.environ["FURUTA_WARM_START_DIR"]) if "FURUTA_WARM_START_DIR" in os.environ else Path(latest_no_dr_run())
WARM_START_MODEL = WARM_START_DIR / "best_model.zip"
WARM_START_NORM = WARM_START_DIR / "vec_normalize_best.pkl"
if not WARM_START_MODEL.exists() or not WARM_START_NORM.exists():
    raise FileNotFoundError(f"Missing warm-start artifacts in {WARM_START_DIR}")

RUN_DIR.mkdir(parents=True, exist_ok=True)
if (RUN_DIR / "run_config.json").exists():
    raise FileExistsError(f"Run directory already contains a training run: {RUN_DIR}")

shutil.copy2(PROJECT_DIR / "furuta_pendulum.xml", RUN_DIR / "furuta_pendulum.xml")
(RUN_DIR / "run_config.json").write_text(
    json.dumps(
        {
            "run_id": RUN_ID,
            "mode": "warm_start_dr",
            "warm_start_run": str(WARM_START_DIR),
            "domain_randomization": True,
            "dr_profile": DR_PROFILE,
            "n_envs": N_ENVS,
            "total_steps": TOTAL_STEPS,
            "curriculum_steps": CURRICULUM_STEPS,
            "episode_seconds": EPISODE_SECONDS,
            "eval_freq": EVAL_FREQ,
            "eval_episodes": EVAL_EPISODES,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "reward_threshold": REWARD_THRESHOLD,
            "disturbance": {
                "elbow_kick_deg": ELBOW_KICK_DEG,
                "elbow_kick_count": ELBOW_KICK_COUNT,
                "timing": "random angle kicks after balance mode, each delayed 1-10 s",
            },
            "reward": {
                "base": "cos(theta) - 0.1*u^2 - 0.5*max(0, abs(phi)-2.094)^2",
                "upright_10deg_bonus": 0.2,
                "upright_5deg_bonus": 0.2,
                "upright_theta_dot_penalty": "-0.005*theta_dot^2 inside 10 deg",
                "post_swingup_fall_rule": f"terminate if outside {FALL_THRESHOLD_DEG:g} deg after first reaching 10 deg",
            },
            "dr": {
                "mass_inertia_range": "+/-5%",
                "shoulder_damping_range": "+/-5%",
                "elbow_damping": "0 to 0.00002 N*m*s/rad in real_ready and real_ready_stage2",
                "motor_torque_scale": "+/-5%",
                "motor_deadband": "0 to 5% in real_ready and real_ready_stage2",
                "action_delay_steps": "0 to 1",
                "shoulder_position_noise_sigma": "0.017 rad at full DR",
                "shoulder_velocity_noise_sigma": "0.050 rad/s at full DR",
                "elbow_position_noise_sigma": "0.001 rad at full DR in real_ready and real_ready_stage2",
                "elbow_velocity": "differentiated from AS5600 position with LP filter alpha in [0.7, 0.9] in real_ready and real_ready_stage2",
                "shoulder_zero_offset": "+/-1 deg in real_ready_stage2",
                "elbow_zero_offset": "+/-0.5 deg in real_ready_stage2",
            },
            "ppo": {
                "ent_coef": ENT_COEF,
                "learning_rate": 1e-4,
                "device": DEVICE,
            },
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


class CurriculumCallback(BaseCallback):
    """Linearly ramp DR progress from 0 to 1 over curriculum_steps timesteps."""

    def __init__(self, schedule: dict, curriculum_steps: int):
        super().__init__()
        self._schedule = schedule
        self._curriculum_steps = curriculum_steps

    def _on_step(self) -> bool:
        self._schedule["progress"] = min(1.0, self.num_timesteps / self._curriculum_steps)
        return True


schedule = {"progress": 0.0}

vec_env = make_vec_env(
    lambda: FurutaPendulumEnv(
        dr_schedule=schedule,
        episode_seconds=EPISODE_SECONDS,
        dr_profile=DR_PROFILE,
        elbow_kick_deg=ELBOW_KICK_DEG,
        elbow_kick_count=ELBOW_KICK_COUNT,
        fall_threshold_deg=FALL_THRESHOLD_DEG,
    ),
    n_envs=N_ENVS,
)
vec_env = VecNormalize.load(str(WARM_START_NORM), vec_env)
vec_env.training = True
vec_env.norm_reward = True

# Evaluate on full DR so the saved best model is selected for robustness, not just nominal performance.
eval_env = make_vec_env(
    lambda: FurutaPendulumEnv(
        domain_rand=True,
        episode_seconds=EPISODE_SECONDS,
        dr_profile=DR_PROFILE,
        elbow_kick_deg=ELBOW_KICK_DEG,
        elbow_kick_count=ELBOW_KICK_COUNT,
        fall_threshold_deg=FALL_THRESHOLD_DEG,
    ),
    n_envs=1,
)
eval_env = VecNormalize.load(str(WARM_START_NORM), eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(
    str(WARM_START_MODEL),
    env=vec_env,
    device=DEVICE,
    ent_coef=ENT_COEF,
    learning_rate=1e-4,
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
    name_prefix="ppo_furuta_dr",
)

curriculum_cb = CurriculumCallback(schedule, curriculum_steps=CURRICULUM_STEPS)

print(f"DR run directory: {RUN_DIR}")
print(f"Warm start      : {WARM_START_DIR}")
print(f"DR profile      : {DR_PROFILE}")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[eval_cb, ckpt_cb, curriculum_cb],
    progress_bar=True,
)

model.save(str(RUN_DIR / "ppo_furuta_dr_final"))
vec_env.save(str(RUN_DIR / "vec_normalize_final.pkl"))

print(f"\nDone. DR run -> {RUN_DIR}")
print(f"Best model  -> {RUN_DIR}/best_model.zip")
print(f"To evaluate -> python eval.py {RUN_DIR}/best_model")
