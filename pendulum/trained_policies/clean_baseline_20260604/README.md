# Clean Baseline Policy

Source run:

```text
runs/no_dr/20260604_163108
```

This policy was trained from scratch in the nominal MuJoCo model without domain
randomization.

Key result:

```text
Best mean reward: 2886.57 / 3000
Episode length: 30 s
Training steps: 3,000,000
```

Use this as the clean reference policy and as the warm-start source for domain
randomization training.

Example warm start:

```powershell
$env:FURUTA_WARM_START_DIR = "trained_policies/clean_baseline_20260604"
$env:FURUTA_DR_PROFILE = "real_ready_stage2"
python train.py
```
