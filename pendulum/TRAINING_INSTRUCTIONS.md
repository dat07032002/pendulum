# Furuta Pendulum Training Instructions

Run all commands from the project pendulum folder:

```powershell
cd C:\Users\thanh\Desktop\Pendulum\main\pendulum
```

## 1. Clean No-DR Training

Train a fresh policy in the nominal MuJoCo model:

```powershell
python train_clean.py
```

Outputs are saved to:

```text
runs/no_dr/<timestamp>/
```

Important files:

```text
best_model.zip
vec_normalize_best.pkl
run_config.json
```

## 2. Hardware-Mass DR Training

Train with the current real-ready domain randomization profile:

```powershell
$env:FURUTA_DR_PROFILE = "real_ready_stage2"
python train.py
```

By default, `train.py` warm-starts from the latest `runs/no_dr/` run.

To warm-start from the Git-tracked hardware-mass DR policy:

```powershell
$env:FURUTA_WARM_START_DIR = "trained_policies/dr_hardware_masses_20260609"
$env:FURUTA_RUN_ID = "my_hardware_dr_run"
$env:FURUTA_DR_PROFILE = "real_ready_stage2"
python train.py
```

Outputs are saved to:

```text
runs/dr/<run_id>/
```

## 3. Deadband and Filter DR Training

The current more hardware-realistic policy adds motor deadband and elbow
velocity filtering:

```text
motor deadband: 0 to 5% of full command
elbow velocity filter alpha: 0.7 to 0.9
```

Continue from the measured-mass DR policy:

```powershell
$env:FURUTA_RUN_ID = "deadband_filter_run"
$env:FURUTA_WARM_START_DIR = "trained_policies/dr_hardware_masses_20260609"
$env:FURUTA_DR_PROFILE = "real_ready_stage2"
$env:FURUTA_EPISODE_SECONDS = "30.0"
$env:FURUTA_ENT_COEF = "0.005"
$env:FURUTA_DEVICE = "cpu"
$env:FURUTA_REWARD_THRESHOLD = "4150"
python train.py
```

The current saved best policy is:

```text
trained_policies/dr_deadband5_filter70_90_20260609
```

## 4. Long-Hold Training

Use the saved Stage 2 or hard-hold model as a warm start, train 60 s episodes,
and use the long-hold reward:

```powershell
$env:FURUTA_RUN_ID = "long_hold_run"
$env:FURUTA_WARM_START_DIR = "runs/dr/20260605_hard_hold_10deg_ent001_stage2"
$env:FURUTA_DR_PROFILE = "real_ready_stage2"
$env:FURUTA_EPISODE_SECONDS = "60.0"
$env:FURUTA_FALL_THRESHOLD_DEG = "20.0"
$env:FURUTA_ENT_COEF = "0.01"
$env:FURUTA_REWARD_THRESHOLD = "7800"
python train.py
```

The long-hold reward adds:

```text
+0.2 if |theta| < 10 deg
+0.2 if |theta| < 5 deg
-0.005 * theta_dot^2 inside 10 deg
terminate after capture if |theta| > FURUTA_FALL_THRESHOLD_DEG
```

Capture starts once the pendulum first reaches `|theta| < 10 deg`.

## 5. Elbow Kick Disturbance Training

Train with small elbow angle disturbances after balance is captured:

```powershell
$env:FURUTA_RUN_ID = "elbow_kick_0p5deg_x3"
$env:FURUTA_WARM_START_DIR = "runs/dr/20260605_hard_hold_10deg_ent001_stage2"
$env:FURUTA_DR_PROFILE = "real_ready_stage2"
$env:FURUTA_EPISODE_SECONDS = "60.0"
$env:FURUTA_ELBOW_KICK_DEG = "0.5"
$env:FURUTA_ELBOW_KICK_COUNT = "3"
$env:FURUTA_FALL_THRESHOLD_DEG = "20.0"
$env:FURUTA_ENT_COEF = "0.01"
$env:FURUTA_REWARD_THRESHOLD = "999999"
python train.py
```

This applies `FURUTA_ELBOW_KICK_COUNT` random elbow angle kicks after balance
mode starts. Each kick is sampled from:

```text
[-FURUTA_ELBOW_KICK_DEG, +FURUTA_ELBOW_KICK_DEG]
```

## 6. Record a GIF

Record a trained model:

```powershell
python record.py runs/dr/<run_id>/best_model.zip runs/dr/<run_id>/eval.gif
```

Example:

```powershell
python record.py runs/dr/20260605_elbow_kick_0p5deg_x3_fall20_longrun/best_model.zip runs/dr/20260605_elbow_kick_0p5deg_x3_fall20_longrun/elbow_kick_best.gif
```

## 7. Git-Tracked Policies

Curated checkpoints are saved in:

```text
trained_policies/
```

Current useful checkpoints:

```text
trained_policies/dr_hardware_masses_20260609
trained_policies/dr_deadband5_filter70_90_20260609
```

Each folder contains `best_model.zip`, `vec_normalize_best.pkl`,
`run_config.json`, the XML snapshot, and evaluation history. These are the
files to push to GitHub. Do not push the full `runs/` folder.

## 8. Notes

- `runs/` is ignored by Git because it contains large generated checkpoints,
  logs, and GIFs.
- Always keep `best_model.zip` paired with its matching
  `vec_normalize_best.pkl`.
- PPO can degrade after a good checkpoint. Use `best_model.zip`, not necessarily
  the final checkpoint.
- For real hardware, calibrate AS5600 upright zero before deployment.
