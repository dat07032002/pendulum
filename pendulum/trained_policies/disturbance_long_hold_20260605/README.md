# Disturbance Long-Hold Policy

Source run:

```text
runs/dr/20260605_elbow_kick_0p5deg_x3_fall20_longrun
```

This policy was trained with 60 s episodes, the long-hold reward, domain
randomization, and small elbow angle disturbances.

Training disturbance:

```text
Elbow kick magnitude: +/-0.5 deg
Kicks per episode: 3
Fall threshold after capture: 20 deg
```

Key result:

```text
Best mean reward: 8233.1 / 8400
Recorded GIF reward: 8240.4 / 8400
Best checkpoint step: about 1.93M
Episode length: 60 s
```

Important note: later PPO updates degraded after the best checkpoint. Use
`best_model.zip`, not a final checkpoint from the same run.
