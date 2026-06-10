# Trained Furuta Pendulum Policies

This folder contains curated checkpoints that are small enough to keep in Git.
The full `runs/` folder is intentionally ignored because it contains generated
logs, GIFs, intermediate checkpoints, and scratch experiments.

Each policy folder contains:

- `best_model.zip` - Stable-Baselines3 PPO checkpoint
- `vec_normalize_best.pkl` - matching `VecNormalize` statistics
- `run_config.json` - training configuration
- `furuta_pendulum.xml` - MuJoCo model snapshot used for the run
- `README.md` - short result summary

Keep `best_model.zip` and `vec_normalize_best.pkl` together. A model loaded
with the wrong normalization statistics can behave badly even if the weights
are correct.

## Hardware Masses

Updated on 2026-06-09:

| Body | Mass |
|---|---:|
| fixed motor / mount | 0.092 kg |
| arm | 0.043 kg |
| pendulum rod | 0.015 kg |

Older policies trained with pre-measurement masses were removed from this
curated folder. Their results remain documented historically in
`documentation.tex`.

## Included Policies

| Folder | Purpose | Best eval | Notes |
|---|---|---:|---|
| `dr_hardware_masses_20260609` | DR baseline with measured masses | 4051.84 / 4200 | Current simple hardware-mass deployment candidate |
| `dr_deadband5_filter70_90_20260609` | DR with measured masses, 0-5% motor deadband, elbow velocity filter alpha 0.7-0.9 | 4032.66 / 4200 | More hardware-realistic candidate; GIF scored 4049.2 / 4200 |

Use the deadband/filter policy when testing robustness to motor dead zone and
AS5600 velocity-estimation lag. Use the hardware-mass policy first if the real
robot needs a simpler initial deployment test.
