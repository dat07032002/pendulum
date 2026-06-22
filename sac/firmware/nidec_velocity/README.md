# Nidec velocity firmware

`nidec_velocity.ino` runs the Furuta arm with an ESP32, an AS5600 pendulum sensor,
and a Nidec motor driver. It includes manual velocity control, on-board balance
control, and motor characterization commands.

## Flash and first-use checklist

1. Open `nidec_velocity.ino` in Arduino IDE with the matching ESP32 board selected.
2. Confirm the motor supply is safe to enable and the arm can move freely within its
   physical ±180° travel.
3. Flash the sketch, open serial at **921600 baud**, and wait for the startup banner.
4. With the motor idle, run `raw`; calibrate with `calhang` (pendulum hanging) or
   `calup` (pendulum exactly upright) only when the reported sensor is healthy.
5. Send `v 2` briefly to confirm direction and `s` to stop. Use `z` only with the
   arm physically centered.

## Safety behavior

- Every motor-enable path, including diagnostics, stops outward motion at ±180°.
  An inward manual command is still allowed after a backstop stop.
- The manual-drive watchdog stops after 200 ms without a command.
- Three failed/rejected AS5600 samples, stale sensor data, invalid magnet status, or
  three consecutive balance-loop deadline misses
  causes a latched motor-disable fault. Fix the sensor, then send `reset`.
- Diagnostic tests accept `s` immediately; they also use their own smaller travel
  limits for repeatability.

## Commands

| Command | Meaning |
| --- | --- |
| `v <rad/s>` | Manual arm velocity from -78 to +78 rad/s. Values below the deadzone stop the motor. |
| `s` | Stop and return to idle. |
| `reset` | Clear a sensor fault only after the AS5600 is healthy. |
| `z` | Zero arm encoder, allowed only while idle. |
| `raw` | Inspect the AS5600 while idle. |
| `calhang`, `calup` | Calibrate the AS5600 while idle and sensor-healthy. |
| `bal` | Arm on-board balance; it engages when the pendulum reaches the configured handoff window. |
| `params` / `?` | Print mode, fault state, and every runtime tuning value. |
| `ksave` | Persist current tuning values to NVS. |
| `defaults` | Restore runtime defaults; follow with `ksave` to make them persistent. |
| `vtest` | Safe no-kick raw-duty sweep: ±20° travel, ±35° hard limit, immediate partial measurements. |
| `dtest`, `ditest`, `ftest` | Run legacy duty, dither, or fine-pulse diagnostics. |

Tuning commands are range-clamped: `k <phi> <theta> <phi_dot> <theta_dot>`, `k1`…`k4`,
`bvmax`, `tr`, `hand`, `handthd`, `kick`, `dgain`, `vkp`, and `vki`.

The firmware uses LQR only. `vkp` and `vki` tune the conservative inner arm-speed PI
correction layered on top of the measured duty-to-speed feed-forward curve.
