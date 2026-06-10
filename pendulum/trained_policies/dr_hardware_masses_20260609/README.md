# Hardware-Mass DR Policy

Source run:

```text
runs/dr/20260609_hardware_masses
```

This policy was warm-started from a clean hardware-mass policy and trained with
the `real_ready_stage2` domain-randomization profile.

Measured masses:

```text
fixed motor / mount: 0.092 kg
arm:                 0.043 kg
pendulum rod:        0.015 kg
```

Result:

```text
Best mean reward: 4051.84 / 4200
Episode length: 30 s
```

This is the simpler measured-mass deployment candidate. It does not include the
new motor deadband and elbow velocity filter randomization.
