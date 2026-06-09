# Trained Furuta Pendulum Policies

This folder contains curated checkpoints that are small enough to keep in Git.
The full `runs/` folder is intentionally ignored because it contains logs,
GIFs, intermediate checkpoints, and generated scratch data.

Each policy folder contains:

- `best_model.zip` - Stable-Baselines3 PPO policy checkpoint
- `vec_normalize_best.pkl` - matching `VecNormalize` statistics
- `run_config.json` - training configuration
- `furuta_pendulum.xml` - MuJoCo model snapshot used for the run
- `evaluations.npz` - evaluation history

Keep `best_model.zip` and `vec_normalize_best.pkl` together. A model loaded
with the wrong normalization statistics can behave badly even if the weights
are correct.

## Included Policies

| Folder | Purpose | Result |
|---|---|---|
| `clean_baseline_20260604` | Clean no-DR policy and warm-start source | 2886.57 / 3000 |
| `domain_randomized_20260604` | Domain-randomized robustness policy | 2554.50 / 3000 |
| `disturbance_long_hold_20260605` | Current best long-hold policy with small elbow kicks | 8233.1 / 8400 |

The disturbance policy is the current best deployment candidate, but the clean
baseline is still useful because DR training warm-starts from it.
