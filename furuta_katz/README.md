# Furuta Pendulum — Ben Katz method, adapted to a velocity-source motor

A fresh, classical-control build of the Furuta pendulum following Ben Katz's
approach (`github.com/bgkatz/Furuta-Pendulum`): **energy-shaping swing-up** plus
**linear/LQR balance**, with a swing-up → handoff → balance state machine.

This is *not* a reinforcement-learning project. It deliberately leaves the SAC
work in `../sac/` behind and rebuilds clean.

## The one adaptation that defines this project

Ben's method assumes the motor is a **torque source** (FOC current control). His
energy pump and his LQR both output a *torque*.

Our motor — **Nidec 24H404H160** — cannot do that. It has an integrated driver
that only accepts a **PWM speed command** (yellow = PWM speed, blue = start/stop,
white = brake). It is a **velocity source**: we command arm *speed*, never torque.
There is no current/torque interface to expose.

So every controller here is the **velocity-source equivalent** of Ben's law:

| Stage | Ben (torque input) | Here (velocity input) |
|---|---|---|
| Swing-up | pump energy with torque: `τ ∝ θ̇·cosθ·dE` | pump energy with arm **acceleration**: `φ̈_des ∝ θ̇·cosθ·dE`, integrated to a speed command `u` |
| Balance | LQR on torque-input linearization | collocated partial-feedback-linearization (arm acceleration as input) + LQR; `u = -K·x` |

This is the principled version of the deadband/min-speed compensation the old
`../sac/lqr_balance.py` used to patch a torque design onto a speed motor.

## Hardware (confirmed)

- **Motor:** Nidec 24H404H160 BLDC, integrated driver, **PWM speed control only**.
  24 V, 25 W, rated torque **38 mN·m**, momentary max **80 mN·m**, ~6000 rpm
  no-load, 12 poles, ball bearings. Arm encoder: **100 lines/rev** (coarse).
- **Pendulum angle (θ):** AS5600 magnetic encoder, 12-bit (4096 CPR). θ = 0 is
  upright.
- **Link:** ESP32 over USB serial @ 921600.
  - ESP32 → PC: `obs=[cos_theta, sin_theta, theta_dot, phi, phi_dot]`
  - PC → ESP32: `u <float>`  (speed command, −1..1), `z` (zero arm encoder)
- **Arm travel:** cable wrap (spring + limited travel), *not* a slip ring. Forces
  a phi limit + recenter logic. A slip ring is the one mechanical upgrade that
  would most closely match Ben's rig (unlimited rotation, no spring).

## Expectations / known limits vs Ben's rig

- 38 mN·m is modest torque → swing-up may need several pumps; tune patiently.
- 100-CPR arm encoder + 4096-CPR θ → noisier velocities than his 20000/16384.
  Keep a velocity observer/filter.
- Target loop rate **200 Hz** (Ben warns balance is hard below ~200 Hz; the old
  rig ran 100 Hz). Verify the ESP32 serial budget supports it.

## Planned files (to build)

- `hw_env.py` — clean serial interface to the ESP32 (obs parse, `u`/`z`, safety).
  Trimmed-down descendant of `../sac/furuta_hw_env.py`.
- `plant.py` — velocity-source Furuta model + linearization about upright
  (arm acceleration as input).
- `swingup.py` — energy-shaping swing-up, velocity-source form.
- `balance.py` — collocated-PFL + LQR balance; `u = -K·x` with velocity observer.
- `run.py` — swing-up → handoff (near upright) → balance state machine.
- `sim.py` — minimal sim to design + validate gains before touching hardware.

## Build order

1. **Plant + sim** — model the velocity-source plant, linearize, confirm upright
   is open-loop unstable and the LQR stabilizes it in sim.
2. **Balance in sim** — design LQR on the velocity-source linearization (no
   deadband hacks), validate region of attraction.
3. **Swing-up in sim** — acceleration-based energy pump; tune handoff band.
4. **Hardware bring-up** — port `hw_env.py`, verify obs + `u` round-trip, push
   loop toward 200 Hz, re-tune velocity filtering.
5. **Balance on hardware**, then **swing-up on hardware**, then tune the handoff.
6. *(Optional mechanical)* add a slip ring to remove the cable spring.

## Source of truth

- Ben Katz repo: https://github.com/bgkatz/Furuta-Pendulum
- His build log: https://build-its-inprogress.blogspot.com/2019/12/furuta-pendulums-building-some-more.html
- Reference port (RL era, to adapt from): `../sac/energy_swingup.py`,
  `../sac/lqr_balance.py`, `../sac/run_swingup_balance.py`, `../sac/furuta_hw_env.py`
