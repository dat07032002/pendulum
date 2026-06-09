# Domain-Randomized Policy

Source run:

```text
runs/dr/20260604_204533_real_ready_stage2
```

This policy was warm-started from the clean baseline and trained with the
domain-randomized profile used for sim-to-real preparation.

Key result:

```text
Best mean reward: 2554.50 / 3000
Episode length: 30 s
Training steps: 3,000,000
```

Robustness comparison under the same randomized evaluation conditions:

```text
Clean policy:             1729.64 / 3000 mean, 60% success
Domain-randomized policy: 2554.67 / 3000 mean, 100% success
```

This is the main result showing that training with randomized dynamics improves
robustness compared with a clean-only policy.
