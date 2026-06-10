# Deadband and Filter DR Policy

Source run:

```text
runs/dr/20260609_deadband5_filter70_90
```

This policy warm-started from `runs/dr/20260609_hardware_masses` and added two
hardware effects:

```text
motor deadband: 0 to 5% of full command
elbow velocity filter alpha: 0.7 to 0.9
```

Result:

```text
Best mean reward: 4032.66 / 4200
Best timestep: 1,297,920
Best std: 31.09
Best mean episode length: 3000 / 3000
Recorded GIF reward: 4049.2 / 4200
Episode length: 30 s
```

This is the more hardware-realistic candidate because it includes motor dead
zone and AS5600 velocity-estimation lag. PPO degraded after the best checkpoint,
so use `best_model.zip`, not later checkpoint files from the raw run folder.
