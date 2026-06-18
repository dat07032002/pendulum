/*
 * Furuta hardware policy firmware — NIDEC brushless servo (integrated driver).
 *
 * Motor control (Nidec 24H @ 12V):
 *   PWM   on D25, INVERTED: duty 0% = full speed, 100% = stop
 *   DIR   on D26: CW/CCW
 *   BRAKE on D27: HIGH = run-enabled, LOW = stopped/disabled
 *   u in [-1,1] -> speed = MIN_SPEED + |u|*(MAX_SPEED-MIN_SPEED), dir = sign(u)
 *     deadband-compensated so the smallest |u| already moves the motor.
 *     (+u -> DIR LOW -> enc increases -> +phi, verified on hardware)
 *
 * Arm encoder (phi): Nidec quadrature CHA/CHB on D34/D35 (5V via level shifter).
 *   ~200 CPR (A-only CHANGE decode).  phi = count * 2*pi / 200.
 *
 * Pendulum (theta): AS5600 on I2C 21/22.  UPRIGHT_RAW = 3977 (hanging = ±180 deg).
 *
 * Serial protocol (unchanged from the brushed firmware):
 *   commands: "u <float>", "s", "z"
 *   stream:   obs=[cos_theta,sin_theta,theta_dot,phi,phi_dot]
 *
 * Safety:
 *   - watchdog: PWM -> stop if no command for WATCHDOG_MS while running
 *   - phi backstop: refuse outward torque beyond +/-120 deg
 *   - MAX_SPEED keeps the (strong) motor gentle until training is tuned
 */

#include <Wire.h>

// --- AS5600 (pendulum theta) ---
#define AS5600_ADDR 0x36
#define AS5600_STATUS 0x0B
#define RAW_ANGLE_H 0x0C
#define AS5600_AGC 0x1A
#define AS5600_MAGNITUDE_H 0x1B
#define SDA_PIN 21
#define SCL_PIN 22
const int UPRIGHT_RAW = 3977;          // recalibrated: hanging read -167.2deg, corrected +12.8deg
const int AS5600_MAX_RAW_STEP = 768;   // reject impossible >67.5deg jumps per sample
const unsigned long AS5600_HEALTH_MS = 500;

// --- Nidec motor control ---
#define PWM_PIN   25                   // inverted: 0% = full, 100% = stop
#define DIR_PIN   26
#define BRAKE_PIN 27                   // HIGH = run-enabled
const int PWM_FREQ = 20000;
const int PWM_RES_BITS = 8;
const int PWM_STOP = 255;              // 100% duty = stop
const float MAX_SPEED = 0.25;          // |u|=1 -> 25% speed
const float MIN_SPEED = 0.06;          // deadband comp: smallest moving speed (~5% measured)

// --- Nidec quadrature encoder (phi) ---
#define ENC_A 34                       // CHA (level-shifted)
#define ENC_B 35                       // CHB
const float SHOULDER_COUNTS_PER_REV = 200.0;

// --- filters / safety ---
const float THETA_VEL_ALPHA = 0.5;   // was 0.25 (too laggy); 0.5 = less theta_dot lag
const float PHI_VEL_ALPHA = 0.85;
const float ACTION_ZERO_ZONE = 0.05;   // 5% measured motor deadband
const unsigned long WATCHDOG_MS = 200;
const float PHI_BACKSTOP_RAD = 120.0 * PI / 180.0;

volatile long shoulder_count = 0;

float theta_prev = 0.0, theta_dot_filt = 0.0;
float phi_prev = 0.0, phi_dot_filt = 0.0, phi_latest = 0.0;
unsigned long t_prev_us = 0;
bool first_sample = true;

float current_u = 0.0;
bool  motor_running = false;
unsigned long last_cmd_ms = 0;

void IRAM_ATTR handleEncoder() {
  int a = digitalRead(ENC_A);
  int b = digitalRead(ENC_B);
  if (a == b) shoulder_count++;
  else        shoulder_count--;
}

uint16_t g_last_raw = 0;
bool g_raw_valid = false;
unsigned long g_i2c_fail = 0;
unsigned long g_raw_jump_reject = 0;
uint8_t g_as5600_status = 0;
uint8_t g_as5600_agc = 0;
uint16_t g_as5600_mag = 0;
bool g_as5600_health_valid = false;
unsigned long g_last_health_ms = 0;

int rawCircularDelta(uint16_t now, uint16_t prev) {
  int diff = (int)now - (int)prev;
  if (diff > 2048) diff -= 4096;
  if (diff < -2048) diff += 4096;
  return diff;
}

bool readAS5600Bytes(uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    g_i2c_fail++;
    return false;
  }
  Wire.requestFrom(AS5600_ADDR, len);
  if (Wire.available() < len) {
    g_i2c_fail++;
    return false;
  }
  for (uint8_t i = 0; i < len; i++) {
    buf[i] = Wire.read();
  }
  return true;
}

void updateAS5600Health() {
  unsigned long now = millis();
  if (now - g_last_health_ms < AS5600_HEALTH_MS) return;
  g_last_health_ms = now;

  uint8_t b[2];
  if (!readAS5600Bytes(AS5600_STATUS, b, 1)) return;
  g_as5600_status = b[0];
  if (!readAS5600Bytes(AS5600_AGC, b, 1)) return;
  g_as5600_agc = b[0];
  if (!readAS5600Bytes(AS5600_MAGNITUDE_H, b, 2)) return;
  g_as5600_mag = ((b[0] & 0x0F) << 8) | b[1];
  g_as5600_health_valid = true;
  // Health printed here (every AS5600_HEALTH_MS), not per-step, to keep the
  // per-step obs line short enough for a high control rate.
  Serial.print("# health as_md="); Serial.print((g_as5600_status & 0x20) ? 1 : 0);
  Serial.print(" as_ml="); Serial.print((g_as5600_status & 0x10) ? 1 : 0);
  Serial.print(" as_mh="); Serial.print((g_as5600_status & 0x08) ? 1 : 0);
  Serial.print(" as_agc="); Serial.print(g_as5600_agc);
  Serial.print(" as_mag="); Serial.print(g_as5600_mag);
  Serial.print(" i2cfail="); Serial.print(g_i2c_fail);
  Serial.print(" rawrej="); Serial.println(g_raw_jump_reject);
}

uint16_t readAS5600Raw() {
  uint8_t b[2];
  if (!readAS5600Bytes(RAW_ANGLE_H, b, 2)) {
    // CRITICAL: never return 0 here -- with UPRIGHT_RAW~27 that reads as
    // ~upright and farms fake reward. Hold the last good value instead.
    return g_raw_valid ? g_last_raw : (uint16_t)(UPRIGHT_RAW + 2048);
  }
  uint16_t candidate = ((b[0] & 0x0F) << 8) | b[1];
  if (g_raw_valid && abs(rawCircularDelta(candidate, g_last_raw)) > AS5600_MAX_RAW_STEP) {
    g_raw_jump_reject++;
    return g_last_raw;
  }
  g_last_raw = candidate;
  g_raw_valid = true;
  return g_last_raw;
}

float wrapToPi(float x) {
  while (x > PI) x -= 2.0 * PI;
  while (x < -PI) x += 2.0 * PI;
  return x;
}

void motorStop() {                     // full stop + disable
  ledcWrite(PWM_PIN, PWM_STOP);
  digitalWrite(BRAKE_PIN, LOW);
  current_u = 0.0;
  motor_running = false;
}

void applyMotorU(float u) {
  if (u > 1.0) u = 1.0;
  if (u < -1.0) u = -1.0;

  // Phi backstop: refuse torque that pushes further past the limit.
  if (fabs(phi_latest) > PHI_BACKSTOP_RAD && u * phi_latest > 0) {
    motorStop();
    Serial.println("# PHI BACKSTOP: outward command refused");
    return;
  }

  if (fabs(u) < ACTION_ZERO_ZONE) {    // zero command: hold, stay enabled
    ledcWrite(PWM_PIN, PWM_STOP);
    digitalWrite(BRAKE_PIN, HIGH);
    current_u = 0.0;
    motor_running = true;
    return;
  }

  digitalWrite(BRAKE_PIN, HIGH);                   // run-enabled
  digitalWrite(DIR_PIN, (u > 0) ? LOW : HIGH);     // +u -> DIR LOW -> +phi
  // Deadband compensation: map |u| in (0,1] onto speed [MIN_SPEED, MAX_SPEED]
  // so the smallest non-zero command already moves the motor (no ~5% dead zone).
  float speed = MIN_SPEED + fabs(u) * (MAX_SPEED - MIN_SPEED);
  if (speed > MAX_SPEED) speed = MAX_SPEED;
  int duty = (int)((1.0 - speed) * 255.0);         // inverted PWM
  if (duty < 0) duty = 0;
  if (duty > 255) duty = 255;
  ledcWrite(PWM_PIN, duty);
  current_u = u;
  motor_running = true;
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();
  if (cmd.length() == 0) return;

  if (cmd == "s") {
    last_cmd_ms = millis();
    motorStop();
    Serial.println("# MOTOR STOP");
    return;
  }
  if (cmd == "z") {
    noInterrupts();
    shoulder_count = 0;
    interrupts();
    phi_prev = 0.0; phi_dot_filt = 0.0; phi_latest = 0.0;
    Serial.println("# Arm encoder zeroed.");
    return;
  }
  if (cmd.startsWith("u")) {
    last_cmd_ms = millis();
    applyMotorU(cmd.substring(1).toFloat());
    return;
  }
  Serial.println("# Unknown command. Use: u 0.25, u -0.25, u 0, s, z");
}

void setup() {
  Serial.begin(921600);   // fast link so the long obs line doesn't cap the rate
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);   // slower I2C = more robust to motor EMI (was 400k)

  pinMode(DIR_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_RES_BITS);
  motorStop();
  last_cmd_ms = millis();

  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), handleEncoder, CHANGE);

  Serial.println("# Furuta NIDEC firmware @ 12V. PWM inverted, BRAKE high=run.");
  Serial.print("# MAX_SPEED="); Serial.print(MAX_SPEED, 2);
  Serial.print(", MIN_SPEED="); Serial.print(MIN_SPEED, 2);
  Serial.print(", CPR="); Serial.print(SHOULDER_COUNTS_PER_REV, 0);
  Serial.print(", phi backstop="); Serial.print(PHI_BACKSTOP_RAD * 180.0 / PI, 0);
  Serial.println(" deg");
  Serial.print("# THETA_VEL_ALPHA="); Serial.print(THETA_VEL_ALPHA, 2);
  Serial.print(", AS5600_MAX_RAW_STEP="); Serial.print(AS5600_MAX_RAW_STEP);
  Serial.println(" counts");
  Serial.println("# Commands: z, u 0.25, u -0.25, u 0, s");
  Serial.println("# obs=[cos_theta,sin_theta,theta_dot,phi,phi_dot] + AS5600 health");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  // Watchdog: enabled but no command recently -> the PC is gone. Fully disable.
  if (motor_running && millis() - last_cmd_ms > WATCHDOG_MS) {
    motorStop();
    Serial.println("# WATCHDOG: no command, motor stopped");
  }

  unsigned long t_now_us = micros();
  float dt = (t_now_us - t_prev_us) * 1e-6;
  if (dt <= 0.0 || dt > 0.1) dt = 0.01;

  // --- theta from AS5600 (hanging = -pi) ---
  uint16_t raw = readAS5600Raw();
  updateAS5600Health();
  float theta = -wrapToPi((raw - UPRIGHT_RAW) * 2.0 * PI / 4096.0);

  // --- phi from Nidec encoder ---
  noInterrupts();
  long count = shoulder_count;
  interrupts();
  float phi = count * 2.0 * PI / SHOULDER_COUNTS_PER_REV;
  phi_latest = phi;

  // Backstop a running motor already past the limit.
  if (motor_running && current_u != 0.0 &&
      fabs(phi) > PHI_BACKSTOP_RAD && current_u * phi > 0) {
    motorStop();
    Serial.println("# PHI BACKSTOP: motor stopped");
  }

  if (first_sample) {
    theta_prev = theta; phi_prev = phi; t_prev_us = t_now_us;
    first_sample = false;
    delay(5);   // ~200 Hz firmware loop -> fresh obs for 100 Hz PC control
    return;
  }

  float dtheta = wrapToPi(theta - theta_prev);
  theta_dot_filt = THETA_VEL_ALPHA * (dtheta / dt) + (1.0 - THETA_VEL_ALPHA) * theta_dot_filt;
  phi_dot_filt   = PHI_VEL_ALPHA   * ((phi - phi_prev) / dt) + (1.0 - PHI_VEL_ALPHA) * phi_dot_filt;

  Serial.print("obs=[");
  Serial.print(cos(theta), 5); Serial.print(",");
  Serial.print(sin(theta), 5); Serial.print(",");
  Serial.print(theta_dot_filt, 5); Serial.print(",");
  Serial.print(phi, 5); Serial.print(",");
  Serial.print(phi_dot_filt, 5); Serial.print("]");
  Serial.print(" theta_deg="); Serial.print(theta * 180.0 / PI, 2);
  Serial.print(" phi_deg="); Serial.print(phi * 180.0 / PI, 2);
  Serial.print(" u="); Serial.print(current_u, 3);
  Serial.print(" i2cfail="); Serial.print(g_i2c_fail);
  Serial.print(" rawrej="); Serial.println(g_raw_jump_reject);

  theta_prev = theta; phi_prev = phi; t_prev_us = t_now_us;
  delay(5);   // ~200 Hz firmware loop -> fresh obs for 100 Hz PC control
}
