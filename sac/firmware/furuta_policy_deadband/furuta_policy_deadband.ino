#include <Wire.h>

#define AS5600_ADDR 0x36
#define RAW_ANGLE_H 0x0C

#define SDA_PIN 21
#define SCL_PIN 22

#define PWMA 25
#define AIN1 26
#define AIN2 27

#define ENC_A 32
#define ENC_B 33

#define UPRIGHT_RAW 2726

const float SHOULDER_COUNTS_PER_REV = 520.0;

const float THETA_VEL_ALPHA = 0.85;
const float PHI_VEL_ALPHA = 0.85;

const int PWM_FREQ = 20000;
const int PWM_RES_BITS = 8;

// Tiny serial noise zone only.
const float ACTION_ZERO_ZONE = 0.005;

// --- Deadband compensation -------------------------------------------------
// Motor ID (motor_pwm_id.csv): torque starts at ~PWM 60 (u ~ 0.235).
// Remap |u| in (0,1] linearly onto PWM [DEADBAND_PWM, 255] so the policy
// gets continuous fine torque starting just above zero. u=0 -> PWM 0.
const float DEADBAND_PWM = 60.0;

// --- Safety: command watchdog ----------------------------------------------
// If no "u" / "s" command arrives within this window while the motor is
// running, cut PWM. The PC sends commands every 20 ms during episodes, so
// this only fires if the PC process dies or the serial link drops.
const unsigned long WATCHDOG_MS = 200;

// --- Safety: arm angle backstop --------------------------------------------
// Independent of the PC: beyond this angle, outward torque is refused and a
// running motor is stopped. Inward (recovery) torque is still allowed.
const float PHI_BACKSTOP_RAD = 120.0 * PI / 180.0;

volatile long shoulder_count = 0;

float theta_prev = 0.0;
float theta_dot_filt = 0.0;

float phi_prev = 0.0;
float phi_dot_filt = 0.0;
float phi_latest = 0.0;

unsigned long t_prev_us = 0;
bool first_sample = true;

float current_u = 0.0;
int current_pwm = 0;

unsigned long last_cmd_ms = 0;

void IRAM_ATTR handleShoulderEncoder() {
  int a = digitalRead(ENC_A);
  int b = digitalRead(ENC_B);

  if (a == b) {
    shoulder_count++;
  } else {
    shoulder_count--;
  }
}

uint16_t readAS5600Raw() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(RAW_ANGLE_H);
  Wire.endTransmission(false);

  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() < 2) return 0;

  uint8_t high = Wire.read();
  uint8_t low = Wire.read();

  return ((high & 0x0F) << 8) | low;
}

float wrapToPi(float x) {
  while (x > PI) x -= 2.0 * PI;
  while (x < -PI) x += 2.0 * PI;
  return x;
}

void motorStop() {
  current_u = 0.0;
  current_pwm = 0;
  ledcWrite(PWMA, 0);
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
}

void applyMotorU(float u) {
  if (u > 1.0) u = 1.0;
  if (u < -1.0) u = -1.0;

  if (fabs(u) < ACTION_ZERO_ZONE) {
    motorStop();
    return;
  }

  // Angle backstop: beyond the limit, refuse torque that pushes further out.
  if (fabs(phi_latest) > PHI_BACKSTOP_RAD && u * phi_latest > 0) {
    motorStop();
    Serial.println("# PHI BACKSTOP: outward command refused");
    return;
  }

  current_u = u;

  // Deadband compensation: |u| in (0,1] -> PWM in [DEADBAND_PWM, 255].
  int pwm = (int)(DEADBAND_PWM + fabs(u) * (255.0 - DEADBAND_PWM));
  if (pwm > 255) pwm = 255;
  if (pwm < 0) pwm = 0;

  current_pwm = pwm;

  if (u > 0) {
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, LOW);
  } else {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, HIGH);
  }

  ledcWrite(PWMA, pwm);
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

    phi_prev = 0.0;
    phi_dot_filt = 0.0;
    phi_latest = 0.0;

    Serial.println("# Shoulder encoder zeroed.");
    return;
  }

  if (cmd.startsWith("u")) {
    last_cmd_ms = millis();
    float u = cmd.substring(1).toFloat();
    applyMotorU(u);
    return;
  }

  Serial.println("# Unknown command. Use: u 0.25, u -0.25, u 0, s, z");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);

  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), handleShoulderEncoder, CHANGE);

  ledcAttach(PWMA, PWM_FREQ, PWM_RES_BITS);
  motorStop();
  last_cmd_ms = millis();

  Serial.println("# Furuta hardware policy firmware: DEADBAND-COMPENSATED motor command");
  Serial.println("# pwm = 60 + |u|*(255-60); watchdog 200ms; phi backstop +/-120deg");
  Serial.println("# STBY must be tied to 3.3V.");
  Serial.println("# Commands: z, u 0.25, u -0.25, u 0, s");
  Serial.println("# obs=[cos_theta,sin_theta,theta_dot,phi,phi_dot]");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  // Watchdog: motor running but no command recently -> the PC is gone.
  if (current_pwm != 0 && millis() - last_cmd_ms > WATCHDOG_MS) {
    motorStop();
    Serial.println("# WATCHDOG: no command, motor stopped");
  }

  unsigned long t_now_us = micros();
  float dt = (t_now_us - t_prev_us) * 1e-6;
  if (dt <= 0.0 || dt > 0.1) dt = 0.01;

  uint16_t raw = readAS5600Raw();

  // Sign convention chosen from real hardware tests.
  float theta = -wrapToPi((raw - UPRIGHT_RAW) * 2.0 * PI / 4096.0);

  noInterrupts();
  long count = shoulder_count;
  interrupts();

  float phi = count * 2.0 * PI / SHOULDER_COUNTS_PER_REV;
  phi_latest = phi;

  // Backstop also against a running motor that is already past the limit.
  if (current_pwm != 0 && fabs(phi) > PHI_BACKSTOP_RAD && current_u * phi > 0) {
    motorStop();
    Serial.println("# PHI BACKSTOP: motor stopped");
  }

  if (first_sample) {
    theta_prev = theta;
    phi_prev = phi;
    t_prev_us = t_now_us;
    first_sample = false;
    delay(10);
    return;
  }

  float dtheta = wrapToPi(theta - theta_prev);
  float theta_dot_raw = dtheta / dt;
  theta_dot_filt = THETA_VEL_ALPHA * theta_dot_raw + (1.0 - THETA_VEL_ALPHA) * theta_dot_filt;

  float phi_dot_raw = (phi - phi_prev) / dt;
  phi_dot_filt = PHI_VEL_ALPHA * phi_dot_raw + (1.0 - PHI_VEL_ALPHA) * phi_dot_filt;

  float cos_theta = cos(theta);
  float sin_theta = sin(theta);

  Serial.print("obs=[");
  Serial.print(cos_theta, 5);
  Serial.print(",");
  Serial.print(sin_theta, 5);
  Serial.print(",");
  Serial.print(theta_dot_filt, 5);
  Serial.print(",");
  Serial.print(phi, 5);
  Serial.print(",");
  Serial.print(phi_dot_filt, 5);
  Serial.print("]");

  Serial.print(" theta_deg=");
  Serial.print(theta * 180.0 / PI, 2);

  Serial.print(" phi_deg=");
  Serial.print(phi * 180.0 / PI, 2);

  Serial.print(" u=");
  Serial.print(current_u, 3);

  Serial.print(" pwm=");
  Serial.println(current_pwm);

  theta_prev = theta;
  phi_prev = phi;
  t_prev_us = t_now_us;

  delay(10);
}
