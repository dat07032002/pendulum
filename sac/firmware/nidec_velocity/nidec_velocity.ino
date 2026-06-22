/*
 * nidec_velocity.ino — velocity-command firmware for the Furuta arm (Nidec 24H @ 12V).
 *
 * Replaces nidec_policy.ino's bang-bang MIN_SPEED map with FINE velocity control,
 * for Ben Katz's classical method (furuta_katz/). The PC controller produces an arm
 * SPEED setpoint v [rad/s] (it integrates the LQR/swing-up acceleration); this
 * firmware maps |v| -> PWM feed-forward from the measured duty->speed curve, then applies
 * a small encoder-based speed PI correction (no MIN_SPEED floor). It also adds a brief
 * break-free kick to beat static stiction at starts/reversals.
 *
 * Measured curve (dtest far-end settled): duty 0.05->2.98, 0.06->4.79, 0.07->8.37,
 * 0.08->10.92, 0.10->18.18 rad/s; duty 0.04 STALLS (min sustained ~3 rad/s).
 * Kick fires ONLY when the arm is physically stopped (measured phi_dot~0), not on every
 * reversal -- so continuous balancing motion is not disturbed.
 *
 * Serial protocol:
 *   commands:  "v <rad/s>" set arm speed | "s" stop | "z" zero arm encoder
 *              "calhang" (pendulum hanging -> sets theta zero, saved to NVS)
 *              "calup"   (pendulum held upright -> sets theta zero) | "raw" (print raw+theta)
 *   stream:    obs=[cos_theta,sin_theta,theta_dot,phi,phi_dot]   (theta=0 upright)
 *              plus a "# health ..." line every 500 ms (kept off the per-step line).
 *
 * Pins (same as nidec_policy.ino):
 *   PWM D25 (invon I2C 21/22 ; Nerted: 0%=full, 100%=stop), DIR D26, BRAKE D27 (HIGH=run)
 *   AS5600 theta idec quadrature CHA/CHB on D34/D35 (~200 CPR)
 *
 * Safety: phi backstop +-180 deg, watchdog stop if no command for WATCHDOG_MS, DUTY_MAX cap.
 */
#include <Wire.h>
#include <Preferences.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

Preferences prefs;   // persists the AS5600 zero across reboots (NVS)

// --- AS5600 (pendulum theta) ---
#define AS5600_ADDR 0x36
#define AS5600_STATUS 0x0B
#define RAW_ANGLE_H 0x0C
#define AS5600_AGC 0x1A
#define AS5600_MAGNITUDE_H 0x1B
#define SDA_PIN 21
#define SCL_PIN 22
int UPRIGHT_RAW = 477;                  // raw AS5600 count at upright; set by calhang/calup, saved to NVS
const int AS5600_MAX_RAW_STEP = 768;
const unsigned long AS5600_HEALTH_MS = 500;
const unsigned long AS5600_STALE_MS = 40;
const uint8_t AS5600_MAX_RAW_FAILURES = 3;
const uint8_t AS5600_STATUS_MAGNET_DETECTED = 0x20;
const uint8_t AS5600_STATUS_MAGNET_TOO_WEAK = 0x10;
const uint8_t AS5600_STATUS_MAGNET_TOO_STRONG = 0x08;

// --- Nidec motor control ---
#define PWM_PIN   25
#define DIR_PIN   26
#define BRAKE_PIN 27
const int PWM_FREQ = 20000;
const int PWM_RES_BITS = 8;
const int PWM_STOP = 255;               // 100% duty = stop (inverted)

// velocity-command tuning (from dtest far-end settled curve)
const float V_DEADZONE = 1.5;           // |v|<this -> hold (min sustained speed ~3 rad/s)
const float DUTY_MAX   = 0.40;          // arm-speed ceiling (swing-up needs a FAST arm)
const float V_COMMAND_MAX = 78.0;       // reject manual commands outside the measured/map range
const float KICK_DUTY  = 0.15;          // break-free pulse to beat stiction
const float DITHER_DUTY = 0.055;        // measured low-duty pulse that permits fine average motion
const unsigned long KICK_MS = 40;       // break-free pulse duration
const float KICK_SPEED = 1.5;           // kick only when |phi_dot| below this (arm stopped)

// duty<->speed curve: measured up to 0.10 (settled), EXTRAPOLATED above (swing-up uses
// the fast end where precision doesn't matter; balance uses the accurate low end).
const int   CURVE_N = 9;
const float CURVE_SPD[CURVE_N] = {2.98, 4.79, 8.37, 10.92, 18.18, 32.0, 45.0, 62.0, 78.0};
const float CURVE_DTY[CURVE_N] = {0.050, 0.060, 0.070, 0.080, 0.100, 0.150, 0.200, 0.300, 0.400};

// --- Nidec quadrature encoder (phi) ---
#define ENC_A 34
#define ENC_B 35
const float SHOULDER_COUNTS_PER_REV = 200.0;

// --- filters / safety ---
const float THETA_VEL_ALPHA = 0.5;
const float PHI_VEL_ALPHA = 0.85;
const unsigned long WATCHDOG_MS = 200;
const float PHI_BACKSTOP_RAD = PI;      // arm bounded to +-180 deg (refuses outward, allows inward)
const unsigned long LOOP_US = 5000;     // 200 Hz target period
const unsigned long DITHER_SLOT_US = LOOP_US;
// The actuator is updated at 200 Hz, so a single dither pulse is exactly one control slot.
// A shorter pulse needs a separate hardware-timed actuator task; do not claim sub-slot timing here.
const unsigned long DITHER_PULSE_US = LOOP_US;

// Runtime-tunable parameters are intentionally bounded. This keeps malformed serial input
// from creating divide-by-zero, inverted limits, or a controller that cannot be stopped safely.
const float BAL_VMAX_MIN = 0.25f, BAL_VMAX_MAX = 20.0f;
const float BAL_TRIM_MAX_RAD = 30.0f * PI / 180.0f;
const float BAL_HANDOFF_MIN_RAD = 0.25f * PI / 180.0f;
const float BAL_HANDOFF_MAX_RAD = 20.0f * PI / 180.0f;
const float BAL_HANDOFF_THD_MAX = 30.0f;
const float DGAIN_MAX = 0.5f, KGAIN_ABS_MAX = 5000.0f;
const float VEL_KP_MAX = 0.05f, VEL_KI_MAX = 0.20f, VEL_I_LIMIT = 10.0f;
const unsigned long BAL_HANDOFF_BLEND_MS = 5;
const uint8_t MAX_CONSECUTIVE_OVERRUNS = 3;

enum RunMode : uint8_t {
  MODE_IDLE,
  MODE_MANUAL,
  MODE_BAL_ARMED,
  MODE_BALANCING,
  MODE_TEST,
  MODE_FAULT
};

// === ON-CHIP BALANCE: LQR + Kalman observer (precomputed in Python @ dt=0.005) ===
// State x=[phi,theta,phi_dot,theta_dot]; measure y=[phi,theta,phi_dot]; estimate theta_dot.
// Running on-chip removes the PC serial latency that made balancing impossible.
const float AD[16]  = {1.000000f,0.000000f,0.005000f,0.000000f, 0.000000f,1.002454f,0.000000f,0.005004f,
                       0.000000f,0.000000f,1.000000f,0.000000f, 0.000000f,0.981802f,0.000000f,1.002454f};
const float BD[4]   = {0.000013f, 0.000009f, 0.005000f, 0.003503f};
const float LOBS[12]= {0.990202f,0.000000f,0.004774f, 0.000000f,1.063394f,0.000000f,
                       0.001066f,0.000000f,0.954451f, 0.000000f,15.010901f,0.000000f};
const float KGAIN_DEFAULT[4] = {-18.31577f, 855.90381f, -9.14419f, 61.18292f};
float Kgain[4] = {-18.31577f, 855.90381f, -9.14419f, 61.18292f};   // LQR gain (tunable via "k a b c d")
const float BAL_A_MAX = 400.0f;         // accel clip
float bal_vmax = 9.0f;                  // balance arm-speed cap [rad/s] (tunable via "bvmax")
float bal_theta_ref = 0.0f;             // balance target offset [rad]; trim so the arm stops drifting ("tr <deg>")
float bal_handoff_rad = 3.0f * PI / 180.0f;   // engage window: within this of upright (tunable "hand <deg>")
float bal_handoff_thd = 1.0f;                 // ...AND slower than this [rad/s] -> a CLEAN catch only
const float BAL_FALL_RAD = 45.0f * PI / 180.0f;   // give up past this
RunMode run_mode = MODE_IDLE;
float xhat[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float vcmd_bal = 0.0f;
float bal_drive_v = 0.0f;               // velocity actually sent through the actuator layer
float bal_prev_applied_a = 0.0f;        // realized velocity change used by observer on the next sample
float bal_handoff_v0 = 0.0f;
unsigned long bal_handoff_started_ms = 0;
const unsigned long BAL_KICK_MS = 18;   // short break-free kick at balance reversals (beat stiction)
bool  bal_was_holding = false;          // arm was in the deadzone last cycle -> kick on restart
bool  bal_kick_on = false;              // enable the reversal kick during balance (toggle "kick 0/1")
// Sub-floor fine speed via DITHER (ditest-proven): below the 3 rad/s motor floor, pulse the
// arm with a GUARANTEED brake after each pulse so it can't coast away. Pulse RATE (delta-
// sigma on |v|) sets the average speed -> the balancer finally gets fine authority near 0.
float bal_dgain = 0.15f;                // on-fraction per rad/s of commanded speed (tune "dgain")
float dith_acc = 0.0f;
bool  dith_pulsed = false;
int   dith_last_dir = 0;
unsigned long dith_slot_started_us = 0;
// Inner speed loop: duty feed-forward from the measured curve plus conservative PI correction.
float vel_kp = 0.0025f;
float vel_ki = 0.0100f;
float vel_i = 0.0f;
int vel_last_dir = 0;

volatile long shoulder_count = 0;

float theta_prev = 0.0, theta_dot_filt = 0.0;
float phi_prev = 0.0, phi_dot_filt = 0.0, phi_latest = 0.0;
unsigned long t_prev_us = 0;
bool first_sample = true;

float current_v = 0.0;                  // commanded arm speed setpoint [rad/s]
bool  motor_moving = false;
unsigned long kick_until = 0;
unsigned long last_cmd_ms = 0;
bool backstop_reported = false;

char command_buf[96];
uint8_t command_len = 0;
unsigned long g_loop_overruns = 0;
unsigned long g_loop_max_us = 0;
uint8_t g_consecutive_overruns = 0;

void IRAM_ATTR handleEncoder() {
  int a = digitalRead(ENC_A);
  int b = digitalRead(ENC_B);
  if (a == b) shoulder_count++;
  else        shoulder_count--;
}

uint16_t g_last_raw = 0;
bool g_raw_valid = false;
unsigned long g_i2c_fail = 0, g_raw_jump_reject = 0;
uint8_t g_as5600_status = 0, g_as5600_agc = 0;
uint16_t g_as5600_mag = 0;
unsigned long g_last_health_ms = 0;
unsigned long g_last_raw_ok_ms = 0;
uint8_t g_raw_fail_streak = 0;
uint8_t g_raw_reject_streak = 0;
bool g_as5600_status_valid = false;
bool g_sensor_fault = false;
const char *g_fault_reason = "none";

int rawCircularDelta(uint16_t now, uint16_t prev) {
  int diff = (int)now - (int)prev;
  if (diff > 2048) diff -= 4096;
  if (diff < -2048) diff += 4096;
  return diff;
}

bool readAS5600Bytes(uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) { g_i2c_fail++; return false; }
  Wire.requestFrom(AS5600_ADDR, len);
  if (Wire.available() < len) { g_i2c_fail++; return false; }
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

bool as5600MagnetHealthy() {
  if (!g_as5600_status_valid) return true;  // status is sampled asynchronously
  return (g_as5600_status & AS5600_STATUS_MAGNET_DETECTED) &&
         !(g_as5600_status & (AS5600_STATUS_MAGNET_TOO_WEAK | AS5600_STATUS_MAGNET_TOO_STRONG));
}

bool as5600DataHealthy() {
  if (!g_raw_valid || g_raw_fail_streak >= AS5600_MAX_RAW_FAILURES || g_raw_reject_streak >= AS5600_MAX_RAW_FAILURES) return false;
  if (millis() - g_last_raw_ok_ms > AS5600_STALE_MS) return false;
  return as5600MagnetHealthy();
}

void updateAS5600Health() {
  unsigned long now = millis();
  if (now - g_last_health_ms < AS5600_HEALTH_MS) return;
  g_last_health_ms = now;
  uint8_t b[2];
  if (!readAS5600Bytes(AS5600_STATUS, b, 1)) return;
  g_as5600_status = b[0];
  g_as5600_status_valid = true;
  if (!readAS5600Bytes(AS5600_AGC, b, 1)) return;
  g_as5600_agc = b[0];
  if (!readAS5600Bytes(AS5600_MAGNITUDE_H, b, 2)) return;
  g_as5600_mag = ((b[0] & 0x0F) << 8) | b[1];
  if (Serial.availableForWrite() < 128) return;
  Serial.print("# health as_agc="); Serial.print(g_as5600_agc);
  Serial.print(" as_mag="); Serial.print(g_as5600_mag);
  Serial.print(" i2cfail="); Serial.print(g_i2c_fail);
  Serial.print(" rawrej="); Serial.println(g_raw_jump_reject);
  Serial.print("# loop_max_us="); Serial.print(g_loop_max_us);
  Serial.print(" overruns="); Serial.println(g_loop_overruns);
}

uint16_t readAS5600Raw() {
  uint8_t b[2];
  if (!readAS5600Bytes(RAW_ANGLE_H, b, 2)) {
    if (g_raw_fail_streak < 255) g_raw_fail_streak++;
    return g_raw_valid ? g_last_raw : (uint16_t)(UPRIGHT_RAW + 2048);
  }
  g_raw_fail_streak = 0;
  uint16_t candidate = ((b[0] & 0x0F) << 8) | b[1];
  if (g_raw_valid && abs(rawCircularDelta(candidate, g_last_raw)) > AS5600_MAX_RAW_STEP) {
    g_raw_jump_reject++;
    if (g_raw_reject_streak < 255) g_raw_reject_streak++;
    return g_last_raw;
  }
  g_raw_reject_streak = 0;
  g_last_raw_ok_ms = millis();
  g_last_raw = candidate;
  g_raw_valid = true;
  return g_last_raw;
}

float wrapToPi(float x) {
  while (x > PI) x -= 2.0 * PI;
  while (x < -PI) x += 2.0 * PI;
  return x;
}

// Direct raw read (circular mean of a few), bypassing the jump-reject filter, for calibration.
// A normal arithmetic average is wrong when samples straddle the AS5600's 0/4095 wrap.
bool calRawRead(uint16_t &result) {
  float sx = 0.0f, sy = 0.0f; int n = 0;
  for (int i = 0; i < 8; i++) {
    uint8_t b[2];
    if (readAS5600Bytes(RAW_ANGLE_H, b, 2)) {
      uint16_t raw = ((b[0] & 0x0F) << 8) | b[1];
      float a = raw * 2.0f * PI / 4096.0f;
      sx += cos(a); sy += sin(a); n++;
    }
    delay(5);
  }
  if (n < 6) return false;
  float a = atan2(sy, sx);
  if (a < 0.0f) a += 2.0f * PI;
  result = (uint16_t)lroundf(a * 4096.0f / (2.0f * PI)) & 0x0FFF;
  return true;
}

void saveUprightRaw(int v) {
  UPRIGHT_RAW = ((v % 4096) + 4096) % 4096;
  prefs.putInt("upraw", UPRIGHT_RAW);
  Serial.print("# UPRIGHT_RAW set to "); Serial.print(UPRIGHT_RAW);
  Serial.println(" (saved to NVS).");
}

// map a desired |v| [rad/s] to raw duty using the measured curve (piecewise linear)
float speedToDuty(float v) {
  if (v <= CURVE_SPD[0]) return CURVE_DTY[0];
  if (v >= CURVE_SPD[CURVE_N - 1]) return CURVE_DTY[CURVE_N - 1];
  for (int i = 0; i < CURVE_N - 1; i++) {
    if (v <= CURVE_SPD[i + 1]) {
      float t = (v - CURVE_SPD[i]) / (CURVE_SPD[i + 1] - CURVE_SPD[i]);
      return CURVE_DTY[i] + t * (CURVE_DTY[i + 1] - CURVE_DTY[i]);
    }
  }
  return CURVE_DTY[CURVE_N - 1];
}

float phiRadNow() {
  noInterrupts();
  long c = shoulder_count;
  interrupts();
  return c * 2.0f * PI / SHOULDER_COUNTS_PER_REV;
}

float clampFloat(float value, float lo, float hi);

bool driveDuty(float frac, int dir) {
  if (run_mode == MODE_FAULT) return false;
  float phi_now = phiRadNow();
  // This is the only motor-enable path, so manual commands and blocking diagnostics
  // all share the same physical arm protection.
  if (fabs(phi_now) >= PHI_BACKSTOP_RAD && dir * phi_now > 0.0f) {
    ledcWrite(PWM_PIN, PWM_STOP);
    digitalWrite(BRAKE_PIN, LOW);
    current_v = 0.0f;
    motor_moving = false;
    run_mode = MODE_IDLE;
    if (!backstop_reported) {
      Serial.println("# PHI BACKSTOP: outward drive disabled; command inward after repositioning");
      backstop_reported = true;
    }
    return false;
  }
  if (dir * phi_now <= 0.0f) backstop_reported = false;
  if (frac > DUTY_MAX) frac = DUTY_MAX;
  if (frac < 0.0) frac = 0.0;
  digitalWrite(BRAKE_PIN, HIGH);
  digitalWrite(DIR_PIN, dir > 0 ? LOW : HIGH);   // +dir -> phi increases
  int duty = (int)((1.0 - frac) * 255.0);
  if (duty < 0) duty = 0;
  if (duty > 255) duty = 255;
  ledcWrite(PWM_PIN, duty);
  return true;
}

void resetVelocityPI() {
  vel_i = 0.0f;
  vel_last_dir = 0;
}

bool driveSpeedPI(float v, float dt) {
  float av = fabs(v);
  int dir = v > 0.0f ? 1 : -1;
  if (dir != vel_last_dir) { vel_i = 0.0f; vel_last_dir = dir; }
  // Project measured arm speed onto the requested direction so reversals receive
  // extra corrective duty until the arm has actually changed direction.
  float measured = dir * phi_dot_filt;
  float error = av - measured;
  float ff = speedToDuty(av);
  float candidate_i = clampFloat(vel_i + error * dt, -VEL_I_LIMIT, VEL_I_LIMIT);
  float duty_unsat = ff + vel_kp * error + vel_ki * candidate_i;
  float duty = clampFloat(duty_unsat, 0.0f, DUTY_MAX);
  // Integrate normally inside the range; at a duty rail, integrate only if that
  // moves the controller back toward the usable range (anti-windup).
  if (duty == duty_unsat || (duty == DUTY_MAX && error < 0.0f) || (duty == 0.0f && error > 0.0f)) vel_i = candidate_i;
  return driveDuty(duty, dir);
}

void motorHold() {                      // zero speed, stay enabled (active brake)
  ledcWrite(PWM_PIN, PWM_STOP);
  digitalWrite(BRAKE_PIN, HIGH);
  resetVelocityPI();
}

void motorStop() {                      // zero speed + disable
  ledcWrite(PWM_PIN, PWM_STOP);
  digitalWrite(BRAKE_PIN, LOW);
  current_v = 0.0;
  motor_moving = false;
  resetVelocityPI();
}

void stopActiveMotion() {
  motorStop();
  vcmd_bal = 0.0f; bal_drive_v = 0.0f; bal_prev_applied_a = 0.0f;
  dith_acc = 0.0f;
  dith_pulsed = false;
  run_mode = MODE_IDLE;
}

void enterFault(const char *reason) {
  if (run_mode != MODE_FAULT) {
    motorStop();
    vcmd_bal = 0.0f; bal_drive_v = 0.0f; bal_prev_applied_a = 0.0f;
    dith_acc = 0.0f;
    dith_pulsed = false;
    run_mode = MODE_FAULT;
    g_fault_reason = reason;
    Serial.print("# FAULT: "); Serial.println(reason);
  }
}

void applyVelocity() {
  float v = current_v;
  // backstop: refuse motion that pushes further past the cable limit
  if (run_mode != MODE_MANUAL || g_sensor_fault) {
    motorStop();
    return;
  }
  if (fabs(v) < V_DEADZONE) { motorHold(); return; }
  int dir = (v > 0) ? 1 : -1;
  // break-free kick ONLY when the arm is physically stopped but commanded to move
  // (keyed on measured speed, not commanded reversals). Continuous motion -> no kick.
  if (fabs(phi_dot_filt) < KICK_SPEED && millis() >= kick_until) {
    kick_until = millis() + KICK_MS;
  }
  if (millis() < kick_until) driveDuty(KICK_DUTY, dir);
  else driveSpeedPI(v, LOOP_US * 1e-6f);
}

bool setVelocity(float v) {
  if (run_mode == MODE_FAULT) { Serial.println("# FAULT active; send reset after sensor recovery"); return false; }
  if (!isfinite(v) || fabs(v) > V_COMMAND_MAX) {
    Serial.print("# velocity must be within +/-"); Serial.println(V_COMMAND_MAX, 1);
    return false;
  }
  current_v = v;
  motor_moving = (fabs(v) >= V_DEADZONE);
  last_cmd_ms = millis();
  run_mode = motor_moving ? MODE_MANUAL : MODE_IDLE;
  if (!motor_moving) motorStop();
  return true;
}

float phiDegNow() {
  return phiRadNow() * 180.0f / PI;
}

// Drive the motor at velocity v [rad/s] for balancing. A SHORT break-free kick fires
// only when restarting from a deadzone hold (a reversal), so the arm doesn't hesitate
// on the stiction and drop the rod -- but continuous motion is left undisturbed.
void driveVel(float v) {
  if (run_mode != MODE_BALANCING) { motorStop(); return; }
  float av = fabs(v);
  int dir = (v > 0) ? 1 : -1;
  if (av < 0.10f) {
    motorHold(); bal_was_holding = true; dith_acc = 0.0f; dith_pulsed = false; dith_last_dir = 0;
    dith_slot_started_us = micros();
    return;
  }
  if (av >= CURVE_SPD[0]) {                 // >= ~3 rad/s: motor sustains it -> continuous drive
    if (bal_kick_on && bal_was_holding) { kick_until = millis() + BAL_KICK_MS; }
    bal_was_holding = false; dith_pulsed = false;
    if (millis() < kick_until) driveDuty(KICK_DUTY, dir);
    else driveSpeedPI(v, LOOP_US * 1e-6f);
    return;
  }
  // 0.10..3 rad/s: one full 5 ms control-slot pulse followed by a guaranteed brake slot.
  // The 1/2 cap prevents consecutive pulse slots without pretending to have a sub-slot timer.
  // Do not make a small correction wait through multiple empty delta-sigma slots.
  // A fresh start or direction reversal receives one immediate *measured low-duty* pulse.
  bool dither_restart = bal_was_holding || dir != dith_last_dir;
  bal_was_holding = false;
  dith_last_dir = dir;
  resetVelocityPI();
  float frac = av * bal_dgain;
  if (frac > 0.5f) frac = 0.5f;
  unsigned long now = micros();
  if (dith_slot_started_us == 0 || now - dith_slot_started_us >= DITHER_SLOT_US) {
    dith_slot_started_us = now;
    if (dither_restart) dith_acc = 1.0f;
    dith_acc += frac;
    dith_pulsed = dith_acc >= 1.0f;
    if (dith_pulsed) dith_acc -= 1.0f;
  }
  if (dith_pulsed && now - dith_slot_started_us < DITHER_PULSE_US) {
    driveDuty(DITHER_DUTY, dir);
  } else {
    motorHold();                            // guaranteed brake after a pulse
  }
}

// One step of on-chip balance. First correct the observer using the saturated, blended
// velocity-command change from the previous tick, then compute the next LQR command.
void balanceStep(float theta, float phi, float phi_dot, float theta_dot_meas, float dt) {
  if (run_mode == MODE_BAL_ARMED) {
    motorHold();
    if (fabs(theta - bal_theta_ref) < bal_handoff_rad && fabs(theta_dot_meas) < bal_handoff_thd) {
      xhat[0] = phi; xhat[1] = theta; xhat[2] = phi_dot; xhat[3] = theta_dot_meas;
      vcmd_bal = phi_dot; bal_drive_v = phi_dot; bal_prev_applied_a = 0.0f;
      bal_handoff_v0 = phi_dot; bal_handoff_started_ms = millis();
      run_mode = MODE_BALANCING;
      Serial.println("# BALANCE engaged");
    }
    return;
  }
  if (fabs(theta - bal_theta_ref) > BAL_FALL_RAD) {
    run_mode = MODE_BAL_ARMED; vcmd_bal = 0.0f; bal_drive_v = 0.0f; bal_prev_applied_a = 0.0f;
    motorHold(); Serial.println("# balance lost -> re-arming");
    return;
  }

  // observer: xhat[k] = AD*xhat[k-1] + BD*a_applied[k-1] + L*(y[k] - C*xhat[k-1])
  float in0 = phi - xhat[0];
  float in1 = wrapToPi(theta - xhat[1]);
  float in2 = phi_dot - xhat[2];
  float xn[4];
  for (int i = 0; i < 4; i++) {
    float ax = AD[i*4]*xhat[0] + AD[i*4+1]*xhat[1] + AD[i*4+2]*xhat[2] + AD[i*4+3]*xhat[3];
    float lx = LOBS[i*3]*in0 + LOBS[i*3+1]*in1 + LOBS[i*3+2]*in2;
    xn[i] = ax + BD[i]*bal_prev_applied_a + lx;
  }
  for (int i = 0; i < 4; i++) xhat[i] = xn[i];

  float th_err = xhat[1] - bal_theta_ref;
  float a = -(Kgain[0]*xhat[0] + Kgain[1]*th_err + Kgain[2]*xhat[2] + Kgain[3]*xhat[3]);
  a = clampFloat(a, -BAL_A_MAX, BAL_A_MAX);
  vcmd_bal = clampFloat(vcmd_bal + a * dt, -bal_vmax, bal_vmax);

  // Blend only the first 5 ms of the catch from measured arm velocity into the LQR command.
  // It preserves the clean handoff while still letting the controller state integrate normally.
  float blend = (millis() - bal_handoff_started_ms) / (float)BAL_HANDOFF_BLEND_MS;
  blend = clampFloat(blend, 0.0f, 1.0f);
  float drive_v = bal_handoff_v0 + blend * (vcmd_bal - bal_handoff_v0);
  bal_prev_applied_a = (drive_v - bal_drive_v) / dt;
  bal_drive_v = drive_v;
  driveVel(drive_v);

  static unsigned long lastBalPrint = 0;
  if (millis() - lastBalPrint > 250 && Serial.availableForWrite() >= 96) {
    lastBalPrint = millis();
    Serial.print("# bal th="); Serial.print(theta * 57.2958f, 1);
    Serial.print(" a="); Serial.print(a, 0);
    Serial.print(" vcmd="); Serial.print(vcmd_bal, 2);
    Serial.print(" drive="); Serial.print(drive_v, 2);
    Serial.print("/"); Serial.println(bal_vmax, 1);
  }
}

// Diagnostics run synchronously, but their stop command must never wait for a newline.
bool testAbortRequested() {
  while (Serial.available()) {
    int ch = Serial.read();
    if (ch == 's' || ch == 'S' || ch == 0x03) {
      stopActiveMotion();
      Serial.println("# test stopped");
      return true;
    }
  }
  return run_mode != MODE_TEST;
}

// Conservative firmware-side breakaway test. It uses raw duty with no kick, drives only
// to +/-8 degrees, then actively brakes. The reported result is transit speed, not a
// settled motor speed: this velocity-source motor reaches breakaway much too abruptly
// for the old long-sweep measurement to be meaningful at low duty.
void runVelTest() {
  // 0.010..0.050 was verified stalled without a kick. This next cautious band finds
  // the actual breakaway threshold without jumping straight to high-speed duty.
  const float DUTY[] = {0.055f, 0.060f, 0.065f, 0.070f};
  const int ND = sizeof(DUTY) / sizeof(DUTY[0]);
  const float STOP_D = 8.0f, MEASURE_LO_D = 1.0f, MEASURE_HI_D = 7.0f, HARD_D = 30.0f;
  const unsigned long PASS_TIMEOUT_MS = 1200;
  Serial.println("# vtest SAFE: raw duty -> transit speed; NO KICK, brake at +-8 deg, hardlimit +-30 deg");
  for (int k = 0; k < ND; k++) {
    float duty = DUTY[k];
    int dir = (phiDegNow() > 0) ? -1 : +1;
    bool in_window = false, reached = false;
    unsigned long entered_ms = 0, t0 = millis();
    float measured = 0.0f;
    while (millis() - t0 < PASS_TIMEOUT_MS) {
      float p = phiDegNow();
      if (fabs(p) > HARD_D) { motorStop(); Serial.println("# vtest ABORT hardlimit"); return; }
      float progress = dir * p;
      if (!driveDuty(duty, dir)) return;
      if (!in_window && progress >= MEASURE_LO_D) { in_window = true; entered_ms = millis(); }
      if (in_window && progress >= MEASURE_HI_D) {
        unsigned long dt = millis() - entered_ms;
        if (dt > 0) measured = ((MEASURE_HI_D - MEASURE_LO_D) * PI / 180.0f) / (dt / 1000.0f);
      }
      if (progress >= STOP_D) { reached = true; break; }
      if (testAbortRequested()) return;
      delay(1);
    }
    motorHold();
    if (!reached) {
      Serial.print("duty="); Serial.print(duty, 3); Serial.println(" result=STALLED_OR_TIMEOUT");
    } else if (measured > 0.0f) {
      Serial.print("duty="); Serial.print(duty, 3);
      Serial.print(" transit="); Serial.print(measured, 2); Serial.println(" rad/s");
    } else {
      Serial.print("duty="); Serial.print(duty, 3); Serial.println(" result=TOO_FAST_TO_TIME");
    }
    delay(400);
  }
  motorStop();
  Serial.println("DONE");
}

// Steady raw-duty test: hold each duty across a long sweep (one kick to start),
// measure the SETTLED center speed far from the kick. Reveals whether low duty
// gives low speed (motor can do fine control) or coasts fast (cannot).
void runDutyTest() {
  const float DTY[] = {0.040, 0.050, 0.060, 0.070, 0.080, 0.100};
  const int ND = 6;
  // Long sweep; measure speed in a window just before the FAR end, after ~0.5 s of
  // pure steady duty (far from the start kick) -> true settled speed, no coast confound.
  const float START_D = 90.0, END_D = 90.0, HARD_D = 105.0;
  const float WIN_LO = 60.0, WIN_HI = 78.0;   // 18 deg window near the far end
  Serial.println("# dtest: raw duty -> settled FAR-END speed (kick only to start, long settle)");
  for (int k = 0; k < ND; k++) {
    float frac = DTY[k];
    float acc = 0.0; int nn = 0;
    int dir = (phiDegNow() > 0) ? -1 : +1;
    unsigned long giveup = millis() + 13000;
    while (nn < 3 && millis() < giveup) {
      unsigned long kick_end = millis() + KICK_MS;     // single kick to break stiction
      bool inwin = false; unsigned long tEnter = 0;
      unsigned long t0 = millis();
      while (millis() - t0 < 6000) {
        float p = phiDegNow();
        if (fabs(p) > HARD_D) { motorStop(); Serial.println("# dtest ABORT hardlimit"); return; }
        float f = (millis() < kick_end) ? KICK_DUTY : frac;   // raw duty, NOT speedToDuty
        if (!driveDuty(f, dir)) return;
        float sp = (dir > 0) ? p : -p;        // progress toward the far end
        if (sp > WIN_LO && !inwin) { inwin = true; tEnter = millis(); }
        else if (sp > WIN_HI && inwin) {
          unsigned long dt = millis() - tEnter;
          if (dt > 0) { acc += ((WIN_HI - WIN_LO) * PI / 180.0) / (dt / 1000.0); nn++; }
          break;
        }
        if (sp >= END_D) break;
        if (testAbortRequested()) return;
        delay(2);
      }
      ledcWrite(PWM_PIN, PWM_STOP);
      delay(200);
      dir = -dir;
    }
    float spd = (nn > 0) ? acc / nn : 0.0;
    Serial.print("duty="); Serial.print(frac, 3);
    Serial.print(" settled="); Serial.print(spd, 2); Serial.println(" rad/s");
  }
  motorStop();
  Serial.println("DONE");
}

// Fine-authority feasibility test: one 5 ms pulse at the measured breakaway duty, followed
// by guaranteed brake slots. It asks whether pulse spacing can produce average arm motion
// below the continuous ~4 rad/s floor, without the long, unsafe sweeps of the old test.
void runDitherTest() {
  const int PULSE_EVERY[] = {8, 6, 4, 3, 2};  // one pulse every N 5 ms slots
  const int NN = sizeof(PULSE_EVERY) / sizeof(PULSE_EVERY[0]);
  const float STOP_D = 8.0f, MEASURE_LO_D = 1.0f, MEASURE_HI_D = 7.0f, HARD_D = 30.0f;
  const unsigned long TEST_TIMEOUT_MS = 5000;
  Serial.println("# ditest SAFE: one 5ms pulse every N slots @ duty .055; brake between; stop +-8 deg, hard +-30 deg");
  for (int k = 0; k < NN; k++) {
    int nslots = PULSE_EVERY[k];
    int dir = (phiDegNow() > 0) ? -1 : +1;
    long slot = 0;
    bool in_window = false, reached = false;
    unsigned long entered_ms = 0, t0 = millis();
    float measured = 0.0f;
    while (millis() - t0 < TEST_TIMEOUT_MS) {
      float p = phiDegNow();
      if (fabs(p) > HARD_D) { motorStop(); Serial.println("# ditest ABORT hardlimit"); return; }
      float progress = dir * p;
      if (slot % nslots == 0) { if (!driveDuty(DITHER_DUTY, dir)) return; }
      else motorHold();
      slot++;
      if (!in_window && progress >= MEASURE_LO_D) { in_window = true; entered_ms = millis(); }
      if (in_window && progress >= MEASURE_HI_D) {
        unsigned long dt = millis() - entered_ms;
        if (dt > 0) measured = ((MEASURE_HI_D - MEASURE_LO_D) * PI / 180.0f) / (dt / 1000.0f);
      }
      if (progress >= STOP_D) { reached = true; break; }
      if (testAbortRequested()) return;
      delay(5);
    }
    motorHold();
    Serial.print("everyN="); Serial.print(nslots);
    Serial.print(" frac="); Serial.print(1.0f / nslots, 3);
    if (!reached) Serial.println(" result=NO_8_DEG_TRAVERSE");
    else if (measured > 0.0f) { Serial.print(" avg_speed="); Serial.print(measured, 2); Serial.println(" rad/s"); }
    else Serial.println(" result=TOO_FAST_TO_TIME");
    delay(600);
  }
  motorStop();
  Serial.println("DONE");
}

// Fine-speed test #2: short SINGLE-cycle pulse with a GUARANTEED brake after every pulse,
// so pulses can never merge/coast (the runaway seen in ditest). Sweep pulse spacing N
// (pulse 1 cycle out of every N). If avg speed ramps smoothly with 1/N down into the
// 0.5-3 rad/s range, fine control IS possible (wire it in). If it still cliffs or stalls,
// the per-pulse momentum is irreducible -> the velocity-source motor is the wall.
void runFineTest() {
  const int   NSP[] = {2, 3, 4, 6, 8, 12};   // pulse 1 cycle out of every N
  const int   NN_ = 6;
  const float HARD_D = 100.0, WIN_LO = 10.0, WIN_HI = 40.0;
  Serial.println("# ftest: 1 short pulse every N cycles (guaranteed brake between) -> avg arm speed");
  for (int k = 0; k < NN_; k++) {
    int N = NSP[k];
    long cyc = 0;
    float acc = 0.0; int nn = 0;
    int dir = (phiDegNow() > 0) ? -1 : +1;
    unsigned long giveup = millis() + 10000;
    while (nn < 2 && millis() < giveup) {
      bool inwin = false; unsigned long tEnter = 0;
      unsigned long t0 = millis();
      while (millis() - t0 < 5000) {
        float p = phiDegNow();
        if (fabs(p) > HARD_D) { motorStop(); Serial.println("# ftest ABORT hardlimit"); return; }
        if (cyc % N == 0) { if (!driveDuty(KICK_DUTY, dir)) return; }  // one short pulse...
        else              motorHold();                 // ...then guaranteed brake
        cyc++;
        float sp = (dir > 0) ? p : -p;
        if (sp > WIN_LO && !inwin) { inwin = true; tEnter = millis(); }
        else if (sp > WIN_HI && inwin) {
          unsigned long dt = millis() - tEnter;
          if (dt > 0) { acc += ((WIN_HI - WIN_LO) * PI / 180.0) / (dt / 1000.0); nn++; }
          break;
        }
        if (sp >= 90.0) break;
        if (testAbortRequested()) return;
        delay(2);                                       // ~2 ms cycle (short pulse)
      }
      ledcWrite(PWM_PIN, PWM_STOP); delay(300);
      dir = -dir; cyc = 0;
    }
    float spd = (nn > 0) ? acc / nn : 0.0;
    Serial.print("everyN="); Serial.print(N);
    Serial.print(" frac="); Serial.print(1.0f / N, 2);
    Serial.print(" avg_speed="); Serial.print(spd, 2); Serial.println(" rad/s");
  }
  motorStop();
  Serial.println("DONE");
}

float clampFloat(float value, float lo, float hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

void normalizeCommand(char *cmd) {
  char *start = cmd;
  while (*start && isspace((unsigned char)*start)) start++;
  if (start != cmd) memmove(cmd, start, strlen(start) + 1);
  size_t len = strlen(cmd);
  while (len > 0 && isspace((unsigned char)cmd[len - 1])) cmd[--len] = '\0';
  for (size_t i = 0; cmd[i]; i++) cmd[i] = (char)tolower((unsigned char)cmd[i]);
}

bool parseFloatArgument(const char *cmd, const char *name, float &value) {
  size_t n = strlen(name);
  if (strncmp(cmd, name, n) != 0 || !isspace((unsigned char)cmd[n])) return false;
  const char *p = cmd + n;
  while (isspace((unsigned char)*p)) p++;
  char *end = NULL;
  value = strtof(p, &end);
  while (end && isspace((unsigned char)*end)) end++;
  return p != end && end && *end == '\0' && isfinite(value);
}

void loadDefaultParams() {
  for (int i = 0; i < 4; i++) Kgain[i] = KGAIN_DEFAULT[i];
  bal_vmax = 9.0f; bal_theta_ref = 0.0f; bal_handoff_rad = 3.0f * PI / 180.0f;
  bal_handoff_thd = 1.0f; bal_kick_on = false; bal_dgain = 0.15f;
  vel_kp = 0.0025f; vel_ki = 0.0100f;
}

void saveParams() {
  for (int i = 0; i < 4; i++) { char key[3] = {'k', (char)('0' + i), '\0'}; prefs.putFloat(key, Kgain[i]); }
  prefs.putFloat("bvmax", bal_vmax); prefs.putFloat("trim", bal_theta_ref);
  prefs.putFloat("hand", bal_handoff_rad); prefs.putFloat("hthd", bal_handoff_thd);
  prefs.putBool("kick", bal_kick_on); prefs.putFloat("dgain", bal_dgain);
  prefs.putFloat("vkp", vel_kp); prefs.putFloat("vki", vel_ki);
}

void printParams() {
  Serial.print("# mode="); Serial.print((int)run_mode);
  Serial.print(" ctrl=LQR");
  Serial.print(" fault="); Serial.print(run_mode == MODE_FAULT ? g_fault_reason : "none");
  Serial.print(" K="); for (int i = 0; i < 4; i++) { Serial.print(Kgain[i], 2); Serial.print(i == 3 ? '|' : ','); }
  Serial.print(" bvmax="); Serial.print(bal_vmax, 2);
  Serial.print(" tr_deg="); Serial.print(bal_theta_ref * 57.2958f, 2);
  Serial.print(" hand_deg="); Serial.print(bal_handoff_rad * 57.2958f, 2);
  Serial.print(" handthd="); Serial.print(bal_handoff_thd, 2);
  Serial.print(" kick="); Serial.print(bal_kick_on ? "ON" : "OFF");
  Serial.print(" dgain="); Serial.print(bal_dgain, 3);
  Serial.print(" vkp="); Serial.print(vel_kp, 4);
  Serial.print(" vki="); Serial.println(vel_ki, 4);
}

// Step-response test: pre-roll the arm at a LOW cruise speed (stiction already broken, no kick),
// then STEP the speed command up and log arm position vs time. The PC reconstructs phi_dot(t)
// and the velocity-loop time constant -- the lag the sim shows is what actually kills balancing.
// Measuring from a moving start (not rest) keeps the stiction kick out of the measurement.
// Prints "t=<ms> tgt=<v> phi=<deg>" during the step.
void runStepTest() {
  const float TGT[] = {6.0f, 10.0f};            // step UP to these...
  const float CRUISE_V = 3.5f;                  // ...from this low cruise (just above the floor)
  const int NT = sizeof(TGT) / sizeof(TGT[0]);
  const float START_D = 165.0f, PREROLL_D = 45.0f, END_D = 160.0f, HARD_D = 175.0f;
  Serial.println("# steptest: step LOW->target while moving -> arm pos vs time (velocity-loop lag)");
  for (int k = 0; k < NT; k++) {
    float vt = TGT[k];
    // 1. drive to the -START_D end (full room to step the +direction)
    unsigned long pos_to = millis() + 4000;
    while (phiDegNow() > -START_D && millis() < pos_to) {
      if (!driveDuty(speedToDuty(5.0f), -1)) return;
      if (testAbortRequested()) return;
      delay(2);
    }
    motorHold();
    delay(400);
    // 2. pre-roll at LOW in +dir (brief kick to break stiction, then steady LOW)
    unsigned long pre_to = millis() + 1500, kick_end = millis() + KICK_MS;
    while (phiDegNow() < -PREROLL_D && millis() < pre_to) {
      float duty = (millis() < kick_end) ? KICK_DUTY : speedToDuty(CRUISE_V);
      if (!driveDuty(duty, +1)) return;
      if (testAbortRequested()) return;
      delay(2);
    }
    // 3. STEP to vt (no kick -- arm is already moving); log position vs time at ~100 Hz
    unsigned long t0 = millis(), t_samp = 0;
    while (millis() - t0 < 650) {
      float p = phiDegNow();
      if (fabs(p) > HARD_D) { motorStop(); Serial.println("# steptest ABORT hardlimit"); return; }
      if (p > END_D) break;                        // ran out of travel
      if (!driveDuty(speedToDuty(vt), +1)) return;
      if (millis() - t_samp >= 10) {
        t_samp = millis();
        Serial.print("t="); Serial.print(millis() - t0);
        Serial.print(" tgt="); Serial.print(vt, 0);
        Serial.print(" phi="); Serial.println(p, 1);
      }
      if (testAbortRequested()) return;
      delay(2);
    }
    motorStop();
    delay(500);
  }
  motorStop();
  Serial.println("DONE");
}

void runSelectedTest(void (*test)()) {
  if (run_mode == MODE_FAULT) { Serial.println("# FAULT active; test refused"); return; }
  motorStop(); vcmd_bal = 0.0f; run_mode = MODE_TEST;
  test();
  if (run_mode == MODE_TEST) { motorStop(); run_mode = MODE_IDLE; }
}

void handleCommand(char *cmd) {
  normalizeCommand(cmd);
  if (*cmd == '\0') return;
  if (!strcmp(cmd, "s")) { stopActiveMotion(); Serial.println("# MOTOR STOP"); return; }
  if (!strcmp(cmd, "reset")) {
    if (!as5600DataHealthy()) { Serial.println("# reset refused: AS5600 is not healthy"); return; }
    g_sensor_fault = false; g_fault_reason = "none"; motorStop(); run_mode = MODE_IDLE; Serial.println("# fault reset; motor idle"); return;
  }
  if (!strcmp(cmd, "vtest")) { runSelectedTest(runVelTest); return; }
  if (!strcmp(cmd, "dtest")) { runSelectedTest(runDutyTest); return; }
  if (!strcmp(cmd, "ditest")) { runSelectedTest(runDitherTest); return; }
  if (!strcmp(cmd, "ftest")) { runSelectedTest(runFineTest); return; }
  if (!strcmp(cmd, "steptest")) { runSelectedTest(runStepTest); return; }
  if (!strcmp(cmd, "bal")) {
    if (run_mode == MODE_FAULT) { Serial.println("# FAULT active; balance refused"); return; }
    motorStop(); vcmd_bal = 0.0f; run_mode = MODE_BAL_ARMED;
    Serial.println("# balance armed: lift the rod upright"); return;
  }
  if (!strcmp(cmd, "?") || !strcmp(cmd, "params")) { printParams(); return; }
  if (!strcmp(cmd, "ksave")) { saveParams(); Serial.println("# parameters saved to NVS"); return; }
  if (!strcmp(cmd, "defaults")) { stopActiveMotion(); loadDefaultParams(); Serial.println("# runtime parameters reset; use ksave to persist"); return; }
  if (!strcmp(cmd, "raw")) {
    if (run_mode != MODE_IDLE) { Serial.println("# raw refused while motion mode is active"); return; }
    uint16_t r;
    if (!calRawRead(r)) { Serial.println("# raw failed: insufficient AS5600 samples"); return; }
    float th = -wrapToPi((r - UPRIGHT_RAW) * 2.0f * PI / 4096.0f);
    Serial.print("# raw="); Serial.print(r); Serial.print(" UPRIGHT_RAW="); Serial.print(UPRIGHT_RAW);
    Serial.print(" theta_deg="); Serial.println(th * 180.0f / PI, 1); return;
  }
  if (!strcmp(cmd, "calhang") || !strcmp(cmd, "calup")) {
    if (run_mode != MODE_IDLE || g_sensor_fault) { Serial.println("# calibration refused until motor is idle and AS5600 is healthy"); return; }
    uint16_t r;
    if (!calRawRead(r)) { Serial.println("# calibration failed: insufficient AS5600 samples"); return; }
    saveUprightRaw(!strcmp(cmd, "calhang") ? r + 2048 : r); return;
  }
  if (!strcmp(cmd, "z")) {
    if (run_mode != MODE_IDLE) { Serial.println("# zero refused while motion mode is active"); return; }
    noInterrupts(); shoulder_count = 0; interrupts();
    phi_prev = 0.0f; phi_dot_filt = 0.0f; phi_latest = 0.0f; Serial.println("# Arm encoder zeroed."); return;
  }
  float value;
  if (parseFloatArgument(cmd, "v", value)) { setVelocity(value); return; }
  if (parseFloatArgument(cmd, "bvmax", value)) { bal_vmax = clampFloat(value, BAL_VMAX_MIN, BAL_VMAX_MAX); Serial.print("# bal_vmax="); Serial.println(bal_vmax, 2); return; }
  if (parseFloatArgument(cmd, "tr", value)) { bal_theta_ref = clampFloat(value * PI / 180.0f, -BAL_TRIM_MAX_RAD, BAL_TRIM_MAX_RAD); Serial.print("# theta_ref deg="); Serial.println(bal_theta_ref * 57.2958f, 2); return; }
  if (parseFloatArgument(cmd, "hand", value)) { bal_handoff_rad = clampFloat(value * PI / 180.0f, BAL_HANDOFF_MIN_RAD, BAL_HANDOFF_MAX_RAD); Serial.print("# handoff deg="); Serial.println(bal_handoff_rad * 57.2958f, 2); return; }
  if (parseFloatArgument(cmd, "handthd", value)) { bal_handoff_thd = clampFloat(value, 0.0f, BAL_HANDOFF_THD_MAX); Serial.print("# handoff_thd="); Serial.println(bal_handoff_thd, 2); return; }
  if (parseFloatArgument(cmd, "kick", value)) { bal_kick_on = value != 0.0f; Serial.print("# bal kick="); Serial.println(bal_kick_on ? "ON" : "OFF"); return; }
  if (parseFloatArgument(cmd, "dgain", value)) { bal_dgain = clampFloat(value, 0.0f, DGAIN_MAX); Serial.print("# dgain="); Serial.println(bal_dgain, 3); return; }
  if (parseFloatArgument(cmd, "vkp", value)) { vel_kp = clampFloat(value, 0.0f, VEL_KP_MAX); resetVelocityPI(); Serial.print("# vel_kp="); Serial.println(vel_kp, 4); return; }
  if (parseFloatArgument(cmd, "vki", value)) { vel_ki = clampFloat(value, 0.0f, VEL_KI_MAX); resetVelocityPI(); Serial.print("# vel_ki="); Serial.println(vel_ki, 4); return; }
  for (int i = 0; i < 4; i++) {
    char key[3] = {'k', (char)('1' + i), '\0'};
    if (parseFloatArgument(cmd, key, value)) { Kgain[i] = clampFloat(value, -KGAIN_ABS_MAX, KGAIN_ABS_MAX); Serial.print("# "); Serial.print(key); Serial.print('='); Serial.println(Kgain[i], 2); return; }
  }
  if (!strncmp(cmd, "k ", 2)) {
    char *p = cmd + 2; float values[4]; bool valid = true;
    for (int i = 0; i < 4; i++) {
      while (isspace((unsigned char)*p)) p++;
      char *end = NULL; values[i] = strtof(p, &end);
      if (p == end || !isfinite(values[i])) { valid = false; break; }
      p = end;
    }
    while (isspace((unsigned char)*p)) p++;
    if (!valid || *p) { Serial.println("# usage: k <phi> <theta> <phi_dot> <theta_dot>"); return; }
    for (int i = 0; i < 4; i++) Kgain[i] = clampFloat(values[i], -KGAIN_ABS_MAX, KGAIN_ABS_MAX);
    printParams(); return;
  }
  Serial.println("# commands: v <rad/s>, s, reset, z, raw, calhang, calup, bal, params, ksave, defaults");
}

void pollSerialCommands() {
  while (Serial.available()) {
    int ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      command_buf[command_len] = '\0'; handleCommand(command_buf); command_len = 0;
    } else if (ch >= 32 && ch <= 126) {
      if (command_len < sizeof(command_buf) - 1) command_buf[command_len++] = (char)ch;
      else { command_len = 0; Serial.println("# command too long"); }
    }
  }
}

void setup() {
  Serial.begin(921600);
  delay(500);
  prefs.begin("furuta", false);
  UPRIGHT_RAW = prefs.getInt("upraw", UPRIGHT_RAW);   // restore saved zero
  loadDefaultParams();
  for (int i = 0; i < 4; i++) { char key[3] = {'k', (char)('0' + i), '\0'}; Kgain[i] = clampFloat(prefs.getFloat(key, Kgain[i]), -KGAIN_ABS_MAX, KGAIN_ABS_MAX); }
  bal_vmax = clampFloat(prefs.getFloat("bvmax", bal_vmax), BAL_VMAX_MIN, BAL_VMAX_MAX);
  bal_theta_ref = clampFloat(prefs.getFloat("trim", bal_theta_ref), -BAL_TRIM_MAX_RAD, BAL_TRIM_MAX_RAD);
  bal_handoff_rad = clampFloat(prefs.getFloat("hand", bal_handoff_rad), BAL_HANDOFF_MIN_RAD, BAL_HANDOFF_MAX_RAD);
  bal_handoff_thd = clampFloat(prefs.getFloat("hthd", bal_handoff_thd), 0.0f, BAL_HANDOFF_THD_MAX);
  bal_kick_on = prefs.getBool("kick", bal_kick_on); bal_dgain = clampFloat(prefs.getFloat("dgain", bal_dgain), 0.0f, DGAIN_MAX);
  vel_kp = clampFloat(prefs.getFloat("vkp", vel_kp), 0.0f, VEL_KP_MAX);
  vel_ki = clampFloat(prefs.getFloat("vki", vel_ki), 0.0f, VEL_KI_MAX);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);                // robust to motor EMI (as in nidec_policy)
  Wire.setTimeOut(5);                   // never let a damaged I2C bus stall the control loop

  pinMode(DIR_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_RES_BITS);
  motorStop();
  last_cmd_ms = millis();

  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), handleEncoder, CHANGE);

  Serial.println("# Furuta NIDEC velocity firmware @ 12V (on-chip balance).");
  Serial.print("# UPRIGHT_RAW="); Serial.println(UPRIGHT_RAW);
  Serial.println("# Safe commands: v <rad/s>, s, reset, z, raw, calhang, calup, bal");
  Serial.println("# LQR tuning: k <4 gains>, ksave, defaults, bvmax/tr/hand/handthd/kick/dgain/vkp/vki");
  Serial.println("# Diagnostics: vtest, dtest, ditest, ftest (send s to stop)");
  Serial.println("# stream: obs=[cos,sin,theta_dot,phi,phi_dot]");
}

void loop() {
  unsigned long loop_start = micros();

  pollSerialCommands();

  if (run_mode == MODE_MANUAL && motor_moving && millis() - last_cmd_ms > WATCHDOG_MS) {
    stopActiveMotion();
    Serial.println("# WATCHDOG: no command, motor stopped");
  }

  unsigned long t_now_us = micros();
  float dt = (t_now_us - t_prev_us) * 1e-6;
  if (dt <= 0.0 || dt > 0.1) dt = 0.005;

  uint16_t raw = readAS5600Raw();
  updateAS5600Health();
  if (!as5600DataHealthy()) {
    g_sensor_fault = true;
    enterFault("AS5600 data stale, unreadable, or magnet status invalid");
  }
  float theta = -wrapToPi((raw - UPRIGHT_RAW) * 2.0 * PI / 4096.0);

  noInterrupts();
  long count = shoulder_count;
  interrupts();
  float phi = count * 2.0 * PI / SHOULDER_COUNTS_PER_REV;
  phi_latest = phi;

  // velocities first -- the on-chip balance controller needs them THIS cycle
  if (!first_sample) {
    float dtheta = wrapToPi(theta - theta_prev);
    theta_dot_filt = THETA_VEL_ALPHA * (dtheta / dt) + (1.0 - THETA_VEL_ALPHA) * theta_dot_filt;
    phi_dot_filt   = PHI_VEL_ALPHA   * ((phi - phi_prev) / dt) + (1.0 - PHI_VEL_ALPHA) * phi_dot_filt;
  }

  // control: on-chip balance (zero serial latency), or the velocity-command path
  if (g_sensor_fault || run_mode == MODE_FAULT) motorStop();
  else if (run_mode == MODE_BAL_ARMED || run_mode == MODE_BALANCING) balanceStep(theta, phi, phi_dot_filt, theta_dot_filt, dt);
  else if (run_mode == MODE_MANUAL) applyVelocity();
  else motorStop();

  // obs every 4th cycle (~50 Hz) so a slow serial reader can't back up the TX buffer and
  // stall the 200 Hz control loop. Control still runs every cycle; only telemetry is decimated.
  static uint8_t obs_div = 0;
  if (!first_sample && (++obs_div & 3) == 0 && Serial.availableForWrite() >= 100) {
    Serial.print("obs=[");
    Serial.print(cos(theta), 5); Serial.print(",");
    Serial.print(sin(theta), 5); Serial.print(",");
    Serial.print(theta_dot_filt, 5); Serial.print(",");
    Serial.print(phi, 5); Serial.print(",");
    Serial.print(phi_dot_filt, 5); Serial.println("]");
  }
  first_sample = false;
  theta_prev = theta; phi_prev = phi; t_prev_us = t_now_us;

  // fixed-period scheduler: hold the loop to LOOP_US (work included, not on top).
  unsigned long elapsed_us = micros() - loop_start;
  if (elapsed_us > g_loop_max_us) g_loop_max_us = elapsed_us;
  if (elapsed_us >= LOOP_US) {
    g_loop_overruns++;
    if (g_consecutive_overruns < 255) g_consecutive_overruns++;
    if (run_mode == MODE_BALANCING && g_consecutive_overruns >= MAX_CONSECUTIVE_OVERRUNS) {
      enterFault("control loop deadline missed repeatedly");
    }
  } else {
    g_consecutive_overruns = 0;
  }
  while (micros() - loop_start < LOOP_US) { /* spin */ }
}
