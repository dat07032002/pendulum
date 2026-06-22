# Furuta Balance Training and Tuning

This project uses a two-controller hardware workflow:

```text
energy swing-up (classical) -> energy-gated handoff -> SAC catch and balance
```

The swing-up controller is tuned on hardware. SAC is trained only for the
upper-region catch and balance task.

## Scripts

| Script | Purpose |
|---|---|
| `calibrate_as5600.py` | Calibrate the upright AS5600 sensor offset. |
| `sysid.py` | Measure motor terminal speed and decay for the simulation model. |
| `test_swingup.py` | Tune hardware pump, coast, centering, and arm damping. |
| `test_swingup_sim.py` | Compare pump gain and coast values in simulation. |
| `train_sac_sim.py` | Train the SAC catch and balance policy. |
| `record_balance_gif.py` | Render a trained balance policy in simulation. |
| `run_swingup_balance.py` | Deploy energy swing-up plus the SAC policy on hardware. |

## Required Order

1. Calibrate `UPRIGHT_RAW` and flash `firmware/nidec_policy/nidec_policy.ino`.
2. Run motor system identification after each firmware `MAX_SPEED` change.
3. Tune hardware swing-up until it repeatedly enters the upper region with
   near-upright energy.
4. Train a clean SAC policy for the same handoff envelope.
5. Warm-start a domain-randomized policy from the clean policy.
6. Deploy with the same handoff values used during training.

## Handoff Contract

The deployment runner hands off from swing-up only when both conditions hold:

```text
abs(theta) < switch-in-deg
abs(E - E_upright) < switch-dE * E_max
```

The defaults are intentionally aligned across deployment and training:

| Item | Default | Why |
|---|---:|---|
| `switch-in-deg` | 30 deg | Upper-region angle accepted for balance. |
| `switch-dE` | 0.08 | Energy-match tolerance as a fraction of `E_max`. |
| `switch-out-deg` | 45 deg | Return to swing-up outside the balance region. Must exceed `switch-in-deg`. |
| `start-angle` | 30 deg | SAC training angle envelope. Must match `switch-in-deg`. |
| `start-vel` | 10 rad/s | Initial catch velocity envelope; verify against measured handoffs. |
| `fall-threshold` | 55 deg | Training termination limit above the balance exit region. |

At 30 degrees, an ideal pendulum with upright-equivalent energy has about
7.25 rad/s of pendulum velocity. The 10 rad/s training limit covers the
energy-gate tolerance and real hardware variation.

Do not deploy a policy outside the angle and velocity envelope it was trained
to catch. If real handoffs consistently occur above 30 degrees, expand both
the deployment and training envelope together, for example to 45 degrees in,
60 degrees out, 45 degrees start angle, and a larger fall threshold.

## Tune Hardware Swing-Up

Start with the pump-and-coast controller only:

```powershell
cd c:\Users\thanh\Desktop\Pendulum\main\sac
python test_swingup.py --port COM5 --k-energy 6 --swingup-umax 0.6 --coast 0.20 --u-floor 0.08 --k-center 0.15 --k-arm-damp 0.02
```

The console reports `theta`, `theta_dot`, `phi`, `dE`, mode, and motor command.

| Symptom | Change |
|---|---|
| Does not reach upper region | Raise `k-energy` or lower `coast`. |
| Blasts through upright | Lower `k-energy` or `swingup-umax`, or raise `coast`. |
| Small commands do nothing | Raise `u-floor` above the firmware zero zone. |
| Arm winds toward one side | Raise `k-center` slightly. |
| Arm is too violent | Raise `k-arm-damp` gradually. |

`coast` disables the energy-pump term near target energy. Centering and arm
damping may still command the arm, so coast does not always mean `u == 0`.

## Train a Clean Catch Policy

Use the voltage/torque motor model and the handoff contract above:

```powershell
cd c:\Users\thanh\Desktop\Pendulum\main\sac
python train_sac_sim.py --total-steps 1500000 --torque --no-domain-rand --curriculum --start-angle 30 --start-vel 10 --fall-threshold 55 --eval-every 20000
```

The evaluator saves `best_model.zip`, `latest_model.zip`, and matching
normalization files under `runs/sac_sim/<timestamp>/`.

Evaluate the candidate before adding randomization. It should catch states
near the edge of the 30-degree and 12-rad/s training envelope, not only easy
near-upright starts.

Render a candidate:

```powershell
python record_balance_gif.py --model-dir runs/sac_sim/<clean_run> --best --start-angle 30 --start-vel 10 --out balance_clean.gif
```

## Warm-Start With Domain Randomization

Start a fresh DR run from clean policy weights. This keeps the learned balance
skill while using a fresh replay buffer and curriculum.

```powershell
python train_sac_sim.py --total-steps 1500000 --torque --curriculum --warm-start runs/sac_sim/<clean_run> --warm-start-best --start-angle 30 --start-vel 10 --fall-threshold 55 --eval-every 20000
```

The simulation randomizes uncertain effects such as delay and sensor behavior.
Motor ranges are narrower because they are grounded by system identification.

## Deploy

Use the trained DR model with the same handoff envelope:

```powershell
python run_swingup_balance.py --model-dir runs/sac_sim/<dr_run> --best --port COM5 --switch-in-deg 30 --switch-out-deg 45 --switch-dE 0.08
```

Keep the hardware action limit equal to the value used during training. If the
hardware needs a smaller limit for safety, train with that same action limit.

## Troubleshooting

| Symptom | Likely Cause | Next Action |
|---|---|---|
| Hand-off never triggers | Energy gate is too tight or swing-up misses target energy. | Raise `switch-dE` slightly or retune swing-up. |
| Immediate return to swing-up | Exit angle is not above entry angle. | Keep `switch-out-deg > switch-in-deg`. |
| Policy catches but does not hold | Arrival states exceed training envelope or DR is too broad. | Lower delivery energy or retrain with a matching wider envelope. |
| Policy saturates | Motor model/action limit mismatch. | Re-run sysid, match action limits, and retrain. |
| Theta looks fake-upright | AS5600 calibration or cable fault. | Recalibrate `UPRIGHT_RAW` and inspect sensor health fields. |
