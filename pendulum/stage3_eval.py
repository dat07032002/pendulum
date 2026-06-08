"""
Stage 3 diagnostic evaluation.

This script does not train. It compares the clean no-DR policy and the
Stage 2 DR policy on:
  1) clean simulation
  2) Stage 2 randomized simulation
  3) a fixed disturbance recovery test

Outputs are written to reports/stage3_eval/.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))

from furuta_env import FurutaPendulumEnv


ROOT = Path(__file__).parent
REPORT_DIR = ROOT / "reports" / "stage3_eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_SECONDS = 30.0
DT = 0.01
N_EPISODES = 20
SEED_BASE = 2000
DISTURBANCE_TIME = 5.0
DISTURBANCE_THETA_DEG = 5.0
DISTURBANCE_THETA_DOT = 1.0
GIF_FPS = 50


@dataclass(frozen=True)
class PolicySpec:
    name: str
    run_dir: Path


@dataclass(frozen=True)
class EvalCondition:
    name: str
    domain_rand: bool
    dr_profile: str
    disturbance: bool = False


POLICIES = [
    PolicySpec("No DR", ROOT / "runs" / "no_dr" / "20260604_163108"),
    PolicySpec("Stage 2 DR", ROOT / "runs" / "dr" / "20260604_204533_real_ready_stage2"),
]

CONDITIONS = [
    EvalCondition("clean", False, "none", False),
    EvalCondition("stage2_randomized", True, "real_ready_stage2", False),
    EvalCondition("disturbance", True, "real_ready_stage2", True),
]


def angle_error(theta: float) -> float:
    return abs(float(np.arctan2(np.sin(theta), np.cos(theta))))


def make_env(policy: PolicySpec, condition: EvalCondition, render: bool = False) -> VecNormalize:
    env = DummyVecEnv([
        lambda: FurutaPendulumEnv(
            render_mode="rgb_array" if render else None,
            domain_rand=condition.domain_rand,
            dr_profile=condition.dr_profile,
            episode_seconds=EPISODE_SECONDS,
        )
    ])
    env = VecNormalize.load(str(policy.run_dir / "vec_normalize_best.pkl"), env)
    env.training = False
    env.norm_reward = False
    return env


def first_lock_time(within_10: np.ndarray) -> float:
    # First time where the pendulum stays inside +/-10 deg for 1 second.
    window = int(round(1.0 / DT))
    for i in range(max(0, len(within_10) - window)):
        if within_10[i:i + window].all():
            return i * DT
    return np.nan


def recovery_time(times: np.ndarray, within_10: np.ndarray) -> float:
    start = int(round(DISTURBANCE_TIME / DT))
    window = int(round(0.5 / DT))
    for i in range(start, max(start, len(within_10) - window)):
        if within_10[i:i + window].all():
            return times[i] - DISTURBANCE_TIME
    return np.nan


def run_episode(
    policy: PolicySpec,
    condition: EvalCondition,
    model: PPO,
    seed: int,
    record: bool = False,
) -> tuple[dict[str, float], list[np.ndarray]]:
    env = make_env(policy, condition, render=record)
    env.seed(seed)
    obs = env.reset()

    frames: list[np.ndarray] = []
    rewards = []
    actions = []
    theta_errs = []
    theta_dots = []
    phis = []
    phi_dots = []
    times = []
    done = [False]
    step = 0
    disturbed = False

    while not done[0]:
        t = step * DT
        if condition.disturbance and not disturbed and t >= DISTURBANCE_TIME:
            env.envs[0].data.qpos[1] += np.deg2rad(DISTURBANCE_THETA_DEG)
            env.envs[0].data.qvel[1] += DISTURBANCE_THETA_DOT
            disturbed = True

        if record:
            frames.append(env.envs[0].render())

        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0, 0]))

        raw_env = env.envs[0]
        theta = float(raw_env.data.qpos[1])
        phi = float(raw_env.data.qpos[0])
        theta_errs.append(angle_error(theta))
        theta_dots.append(float(raw_env.data.qvel[1]))
        phis.append(phi)
        phi_dots.append(float(raw_env.data.qvel[0]))
        times.append(t)

        obs, reward, done, _info = env.step(action)
        rewards.append(float(reward[0]))
        step += 1

    env.close()

    theta_errs_a = np.asarray(theta_errs)
    theta_dots_a = np.asarray(theta_dots)
    phi_dots_a = np.asarray(phi_dots)
    phis_a = np.asarray(phis)
    actions_a = np.asarray(actions)
    rewards_a = np.asarray(rewards)
    times_a = np.asarray(times)
    within_10 = theta_errs_a < np.deg2rad(10)
    within_5 = theta_errs_a < np.deg2rad(5)

    after_swing = times_a >= 2.0
    if not np.any(after_swing):
        after_swing = np.ones_like(times_a, dtype=bool)

    metrics = {
        "reward": float(np.sum(rewards_a)),
        "time_within_10_deg": float(np.sum(within_10) * DT),
        "time_within_5_deg": float(np.sum(within_5) * DT),
        "mean_angle_error_deg": float(np.degrees(np.mean(theta_errs_a))),
        "mean_angle_error_after_2s_deg": float(np.degrees(np.mean(theta_errs_a[after_swing]))),
        "max_angle_error_deg": float(np.degrees(np.max(theta_errs_a))),
        "theta_dot_rms_after_2s": float(np.sqrt(np.mean(theta_dots_a[after_swing] ** 2))),
        "phi_dot_rms_after_2s": float(np.sqrt(np.mean(phi_dots_a[after_swing] ** 2))),
        "action_rms": float(np.sqrt(np.mean(actions_a ** 2))),
        "control_energy": float(np.sum(actions_a ** 2) * DT),
        "shoulder_range_deg": float(np.degrees(np.max(phis_a) - np.min(phis_a))),
        "swingup_time_s": float(first_lock_time(within_10)),
        "success": float(np.sum(within_10) * DT > 20.0),
        "recovery_time_s": float(recovery_time(times_a, within_10)) if condition.disturbance else np.nan,
    }
    return metrics, frames


def summarize(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    metric_names = [
        "reward",
        "time_within_10_deg",
        "time_within_5_deg",
        "mean_angle_error_after_2s_deg",
        "theta_dot_rms_after_2s",
        "action_rms",
        "control_energy",
        "shoulder_range_deg",
        "swingup_time_s",
        "success",
        "recovery_time_s",
    ]
    summary = []
    for policy in sorted({str(r["policy"]) for r in rows}):
        for condition in sorted({str(r["condition"]) for r in rows}):
            subset = [r for r in rows if r["policy"] == policy and r["condition"] == condition]
            if not subset:
                continue
            out: dict[str, str | float] = {"policy": policy, "condition": condition, "episodes": len(subset)}
            for metric in metric_names:
                vals = np.asarray([float(r[metric]) for r in subset], dtype=float)
                finite = vals[np.isfinite(vals)]
                if len(finite) == 0:
                    out[f"{metric}_mean"] = np.nan
                    out[f"{metric}_std"] = np.nan
                    out[f"{metric}_min"] = np.nan
                    continue
                out[f"{metric}_mean"] = float(np.mean(finite))
                out[f"{metric}_std"] = float(np.std(finite))
                out[f"{metric}_min"] = float(np.min(finite))
            summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: list[dict[str, str | float]]) -> None:
    conditions = ["clean", "stage2_randomized", "disturbance"]
    policies = ["No DR", "Stage 2 DR"]
    colors = {"No DR": "#4C78A8", "Stage 2 DR": "#F58518"}

    def get(policy: str, condition: str, metric: str) -> float:
        for row in summary_rows:
            if row["policy"] == policy and row["condition"] == condition:
                return float(row[metric])
        return np.nan

    x = np.arange(len(conditions))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
    axes = axes.ravel()
    plot_defs = [
        ("reward_mean", "Mean reward", "Reward"),
        ("time_within_10_deg_mean", "Time within +/-10 deg", "Seconds"),
        ("mean_angle_error_after_2s_deg_mean", "Mean angle error after 2 s", "Degrees"),
        ("shoulder_range_deg_mean", "Shoulder range", "Degrees"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, plot_defs):
        for j, policy in enumerate(policies):
            vals = [get(policy, c, metric) for c in conditions]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=policy, color=colors[policy])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean", "Stage 2", "Disturbance"], rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "stage3_balance_metrics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows: list[dict[str, str | float]] = []

    for policy in POLICIES:
        model = PPO.load(str(policy.run_dir / "best_model"), device="cpu")
        for condition in CONDITIONS:
            for episode in range(N_EPISODES):
                seed = SEED_BASE + episode
                metrics, _frames = run_episode(policy, condition, model, seed, record=False)
                row: dict[str, str | float] = {
                    "policy": policy.name,
                    "condition": condition.name,
                    "episode": episode,
                    "seed": seed,
                }
                row.update(metrics)
                rows.append(row)
                print(
                    f"{policy.name:10s} {condition.name:18s} ep={episode:02d} "
                    f"reward={metrics['reward']:.1f} within10={metrics['time_within_10_deg']:.1f}s"
                )

    summary_rows = summarize(rows)
    write_csv(REPORT_DIR / "stage3_eval_metrics.csv", rows)
    write_csv(REPORT_DIR / "stage3_eval_summary.csv", summary_rows)
    plot_summary(summary_rows)

    # Representative disturbance GIFs, one per policy, using the first seed.
    for policy in POLICIES:
        model = PPO.load(str(policy.run_dir / "best_model"), device="cpu")
        condition = EvalCondition("disturbance", True, "real_ready_stage2", True)
        metrics, frames = run_episode(policy, condition, model, SEED_BASE, record=True)
        safe_name = policy.name.lower().replace(" ", "_")
        gif_path = REPORT_DIR / f"{safe_name}_disturbance_recovery.gif"
        imageio.mimsave(str(gif_path), frames, fps=GIF_FPS, loop=0)
        print(f"saved {gif_path} reward={metrics['reward']:.1f}")

    print(f"wrote {REPORT_DIR / 'stage3_eval_metrics.csv'}")
    print(f"wrote {REPORT_DIR / 'stage3_eval_summary.csv'}")
    print(f"wrote {REPORT_DIR / 'stage3_balance_metrics.png'}")


if __name__ == "__main__":
    main()
