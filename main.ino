// ======================================================
// main.ino — control flow, globals, tuning, debug
// Motor x is black, y is white
// ======================================================

#include <Wire.h>
#include <EEPROM.h>

#define MPU6050       0x68
#define ACCEL_CONFIG  0x1C
#define GYRO_CONFIG   0x1B
#define PWR_MGMT_1    0x6B

// Motor pins
#define PWM_X          10
#define DIRECTION_X    6

#define PWM_Y          9
#define DIRECTION_Y    4

// Encoder pins
#define ENC_X_A        2
#define ENC_X_B        3

#define ENC_Y_A        7
#define ENC_Y_B        8

// Nidec 24H: 100 PPR encoder × 4x quadrature decode = 400 counts per revolution
const float PPR_X = 100.0;
const float PPR_Y = 100.0;

const int MAX_PWM_CMD   = 150;
const int K3_PWM_LIMIT  = 60;

const unsigned long SPEED_UPDATE_US = 10000;
const float SPEED_ALPHA = 0.35; // 0.2 - 0.5 

const unsigned long K3_DELAY_MS = 200;
const unsigned long K3_RAMP_MS  = 400;

bool wasBalancing = false;
unsigned long balanceStartTime = 0;
float k3Scale = 0.0;

// Relay auto-tune (Åström-Hägglund) — one axis at a time
bool relayTuningX = false;
bool relayTuningY = false;
const int           RELAY_D            = 40;     // relay amplitude (PWM units)
const float         RELAY_K2           = 6.0f;   // damping during relay test
const unsigned long RELAY_DURATION_MS  = 5000;

unsigned long relay_startTime    = 0;
unsigned long relay_lastCross_us = 0;
float relay_periodSum = 0.0f;
int   relay_periodN   = 0;
float relay_maxAngle  = 0.0f;
float relay_minAngle  = 0.0f;
float relay_prevAngle = 0.0f;

// X controller gains
float K1_X = 50.0;
float KI_X = 0.0;
float K2_X = 6.0;
float K3_X = 0.0;

// Y controller gains
float K1_Y = 50.0;
float KI_Y = 0.0;
float K2_Y = 6.0;
float K3_Y = 0.0;

float angleX_integral = 0.0;
float angleY_integral = 0.0;
const float INTEGRAL_LIMIT = 5.0;

float loop_time_us = 1000.0;
float loop_time_s  = 0.001;
float alpha = 0.15;

struct OffsetsObj {
  int ID;
  float X;
  float Y;
};
OffsetsObj offsets;

int pwm_X = 0;
int pwm_Y = 0;

float k3_pwm_X = 0.0;
float k3_pwm_Y = 0.0;

volatile long encoderCountX = 0;
volatile long encoderCountY = 0;

float wheelSpeedX_cps = 0.0;
float wheelSpeedY_cps = 0.0;

long lastEncoderCountX = 0;
long lastEncoderCountY = 0;
unsigned long lastSpeedTime_us = 0;

unsigned long currentT_us    = 0;
unsigned long previousT_1_us = 0;
unsigned long previousT_2_us = 0;

int16_t AcX, AcY, AcZ;
int32_t GyY, GyZ;

#define accSens     0
#define gyroSens    1
#define Gyro_amount 0.996

int16_t AcX_offset = 0;
int16_t AcY_offset = 0;
int16_t AcZ_offset = 0;

int16_t GyY_offset = 0;
int16_t GyZ_offset = 0;
int32_t GyY_offset_sum = 0;
int32_t GyZ_offset_sum = 0;

float robot_angleX = 0.0;
float robot_angleY = 0.0;
float angleX = 0.0;
float angleY = 0.0;
float Acc_angleX = 0.0;
float Acc_angleY = 0.0;

float gyroZ = 0.0;
float gyroY = 0.0;
float gyroZfilt = 0.0;
float gyroYfilt = 0.0;

bool vertical    = false;
bool calibrating = false;
bool calibrated  = false;

// ======================================================
// K3 ramp
// ======================================================
float getK3Scale() {
  unsigned long t = millis() - balanceStartTime;
  if (t < K3_DELAY_MS) return 0.0;
  if (t < K3_DELAY_MS + K3_RAMP_MS)
    return (float)(t - K3_DELAY_MS) / (float)K3_RAMP_MS;
  return 1.0;
}

// ======================================================
// Relay auto-tune (Åström-Hägglund)
//
// While active on an axis, the controller for that axis is replaced by
//   u = d * sign(angle) + RELAY_K2 * gyro
// which forces a sustained limit-cycle oscillation. We measure:
//   Tu  = period of oscillation
//   a   = half peak-to-peak amplitude of angle
//   Ku  = 4*d / (pi*a)         (describing-function approximation)
// then print Ziegler-Nichols PID suggestions.
//
// The other axis keeps balancing normally.
// ======================================================
void relayResetState() {
  relay_startTime    = millis();
  relay_lastCross_us = 0;
  relay_periodSum    = 0.0f;
  relay_periodN      = 0;
  relay_maxAngle     = 0.0f;
  relay_minAngle     = 0.0f;
  relay_prevAngle    = 0.0f;
}

void startRelayTuneX() {
  if (relayTuningX || relayTuningY) { Serial.println(F("Relay already running.")); return; }
  if (!vertical || !calibrated)     { Serial.println(F("Must be balancing first."));  return; }
  relayTuningX = true;
  relayResetState();
  Serial.println(F("Relay tuning X for 5s — pendulum WILL wobble..."));
}

void startRelayTuneY() {
  if (relayTuningX || relayTuningY) { Serial.println(F("Relay already running.")); return; }
  if (!vertical || !calibrated)     { Serial.println(F("Must be balancing first."));  return; }
  relayTuningY = true;
  relayResetState();
  Serial.println(F("Relay tuning Y for 5s — pendulum WILL wobble..."));
}

void relayPrintResults(char axis) {
  if (relay_periodN < 4) {
    Serial.print(F("Relay ")); Serial.print(axis);
    Serial.println(F(": too few zero-crossings. Try larger RELAY_D or longer run."));
    return;
  }
  float Tu = relay_periodSum / (float)relay_periodN;
  float a  = (relay_maxAngle - relay_minAngle) * 0.5f;
  if (a < 0.1f) { Serial.println(F("Amplitude too small — invalid Ku.")); return; }

  float Ku = 4.0f * (float)RELAY_D / (3.14159f * a);

  Serial.println();
  Serial.print(F("=== Relay tune ")); Serial.print(axis); Serial.println(F(" result ==="));
  Serial.print(F("  Ku  = ")); Serial.println(Ku, 2);
  Serial.print(F("  Tu  = ")); Serial.print(Tu * 1000.0f, 1); Serial.println(F(" ms"));
  Serial.print(F("  amp = ")); Serial.print(a, 2); Serial.println(F(" deg"));
  Serial.print(F("  N   = ")); Serial.println(relay_periodN);
  Serial.println(F("Ziegler-Nichols PID suggestions:"));
  if (axis == 'X') {
    Serial.print(F("  K1_X = ")); Serial.println(0.6f   * Ku,      2);
    Serial.print(F("  KI_X = ")); Serial.println(1.2f   * Ku / Tu, 3);
    Serial.print(F("  K2_X = ")); Serial.println(0.075f * Ku * Tu, 3);
    Serial.println(F("Apply:  p <val>   o <val>   i <val>"));
  } else {
    Serial.print(F("  K1_Y = ")); Serial.println(0.6f   * Ku,      2);
    Serial.print(F("  KI_Y = ")); Serial.println(1.2f   * Ku / Tu, 3);
    Serial.print(F("  K2_Y = ")); Serial.println(0.075f * Ku * Tu, 3);
    Serial.println(F("Apply:  P <val>   O <val>   I <val>"));
  }
  Serial.println(F("=========================="));
}

// Called once per 1 ms control tick while relayTuningX is true.
void relayUpdateX() {
  if (angleX > relay_maxAngle) relay_maxAngle = angleX;
  if (angleX < relay_minAngle) relay_minAngle = angleX;

  // Zero-crossing detection (sign change)
  if ((relay_prevAngle <= 0.0f && angleX > 0.0f) ||
      (relay_prevAngle >= 0.0f && angleX < 0.0f)) {
    if (relay_lastCross_us != 0) {
      // Time between two consecutive crossings = Tu/2, so multiply by 2.
      relay_periodSum += (currentT_us - relay_lastCross_us) * 2e-6f;
      relay_periodN++;
    }
    relay_lastCross_us = currentT_us;
  }
  relay_prevAngle = angleX;

  if (millis() - relay_startTime >= RELAY_DURATION_MS) {
    relayTuningX = false;
    relayPrintResults('X');
  }
}

void relayUpdateY() {
  if (angleY > relay_maxAngle) relay_maxAngle = angleY;
  if (angleY < relay_minAngle) relay_minAngle = angleY;

  if ((relay_prevAngle <= 0.0f && angleY > 0.0f) ||
      (relay_prevAngle >= 0.0f && angleY < 0.0f)) {
    if (relay_lastCross_us != 0) {
      relay_periodSum += (currentT_us - relay_lastCross_us) * 2e-6f;
      relay_periodN++;
    }
    relay_lastCross_us = currentT_us;
  }
  relay_prevAngle = angleY;

  if (millis() - relay_startTime >= RELAY_DURATION_MS) {
    relayTuningY = false;
    relayPrintResults('Y');
  }
}

// ======================================================
// Setup
// ======================================================
void setup() {
  Serial.begin(115200);

  pinMode(DIRECTION_X, OUTPUT);
  pinMode(DIRECTION_Y, OUTPUT);
  pinMode(PWM_X, OUTPUT);
  pinMode(PWM_Y, OUTPUT);

  analogWriteResolution(8);
  analogWriteFrequency(PWM_X, 20000);
  analogWriteFrequency(PWM_Y, 20000);

  analogWrite(PWM_X, 255);
  analogWrite(PWM_Y, 255);

  setupEncoders();
  delay(1000);

  EEPROM.get(0, offsets);
  if (offsets.ID == 33) {
    calibrated = true;
  } else {
    calibrated = false;
    offsets.ID = 0;
    offsets.X  = 0.0f;
    offsets.Y  = 0.0f;
  }

  Serial.println(F("Starting IMU calibration — hold robot vertical and still..."));
  angle_setup();

  //Serial.println();
  //Serial.println(F("Commands:"));
  //Serial.println(F("p+ / p- : K1_X"));
  //Serial.println(F("o+ / o- : KI_X"));
  //Serial.println(F("i+ / i- : K2_X"));
  //Serial.println(F("s+ / s- : K3_X"));
  //Serial.println(F("P+ / P- : K1_Y"));
  //Serial.println(F("O+ / O- : KI_Y"));
  //Serial.println(F("I+ / I- : K2_Y"));
  //Serial.println(F("S+ / S- : K3_Y"));
  //Serial.println(F("c+      : calibration ON"));
  //Serial.println(F("c-      : save balance point"));
  //Serial.println();

  lastSpeedTime_us = micros();
  previousT_1_us   = micros();
  previousT_2_us   = micros();

  printValues();
}

// ======================================================
// Main loop
// ======================================================
void loop() {
  currentT_us = micros();

  // 1 ms control loop
  if (currentT_us - previousT_1_us >= (unsigned long)loop_time_us) {

    loop_time_s = (currentT_us - previousT_1_us) * 1e-6f;

    Tuning();
    updateEncoderSpeed();

    angle_calc();

    gyroZ = GyZ / 65.536f;
    gyroY = GyY / 65.536f;
    gyroZfilt = alpha * gyroZ + (1.0f - alpha) * gyroZfilt;
    gyroYfilt = alpha * gyroY + (1.0f - alpha) * gyroYfilt;

    if (vertical && calibrated) {
      if (!wasBalancing) {
        balanceStartTime = millis();
        wasBalancing = true;
      }

      k3Scale = getK3Scale();

      angleX_integral += angleX * loop_time_s;
      angleY_integral += angleY * loop_time_s;
      angleX_integral = constrain(angleX_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);
      angleY_integral = constrain(angleY_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

      k3_pwm_X = constrain(-k3Scale * K3_X * wheelSpeedX_cps, -K3_PWM_LIMIT, K3_PWM_LIMIT);
      k3_pwm_Y = constrain(-k3Scale * K3_Y * wheelSpeedY_cps, -K3_PWM_LIMIT, K3_PWM_LIMIT);

      // X axis — relay test or normal control
      if (relayTuningX) {
        pwm_X = constrain(
          (angleX > 0.0f ? RELAY_D : -RELAY_D) + (int)(RELAY_K2 * gyroZfilt),
          -MAX_PWM_CMD, MAX_PWM_CMD);
        relayUpdateX();
      } else {
        pwm_X = constrain(
          K1_X * angleX + KI_X * angleX_integral + K2_X * gyroZfilt + k3_pwm_X,
          -MAX_PWM_CMD, MAX_PWM_CMD);
      }

      // Y axis — relay test or normal control
      if (relayTuningY) {
        pwm_Y = constrain(
          (angleY > 0.0f ? RELAY_D : -RELAY_D) + (int)(RELAY_K2 * gyroYfilt),
          -MAX_PWM_CMD, MAX_PWM_CMD);
        relayUpdateY();
      } else {
        pwm_Y = constrain(
          K1_Y * angleY + KI_Y * angleY_integral + K2_Y * gyroYfilt + k3_pwm_Y,
          -MAX_PWM_CMD, MAX_PWM_CMD);
      }

      if (!calibrating) {
        Motor_controlX(pwm_X);
        Motor_controlY(pwm_Y);
      } else {
        Motor_controlX(0);
        Motor_controlY(0);
      }

    } else {
      Motor_controlX(0);
      Motor_controlY(0);

      wheelSpeedX_cps = 0.0f;
      wheelSpeedY_cps = 0.0f;
      k3_pwm_X = 0.0f;
      k3_pwm_Y = 0.0f;

      angleX_integral = 0.0f;
      angleY_integral = 0.0f;

      wasBalancing = false;
      k3Scale = 0.0f;

      if (relayTuningX || relayTuningY) {
        Serial.println(F("Lost balance — relay tune aborted."));
        relayTuningX = false;
        relayTuningY = false;
      }
    }

    previousT_1_us = currentT_us;
  }

  // 500 ms debug print
  if (currentT_us - previousT_2_us >= 500000UL) {
    printDebug();
    if (!calibrated && !calibrating)
      Serial.println(F("First calibrate balance point with c+ then c-"));
    previousT_2_us = currentT_us;
  }
}

// ======================================================
// Serial tuning
// Commands (one per line, terminated by Enter):
//   p <val>   set K1_X    (e.g. "p 90")
//   o <val>   set KI_X    (e.g. "o 0.1")
//   i <val>   set K2_X    (e.g. "i 6.5")
//   s <val>   set K3_X    (e.g. "s 0.005")
//   P/O/I/S   same for Y axis (uppercase)
//   c+        start balance calibration
//   c-        save balance calibration
//   ?         print current gains
// Increments still work too:  "p+" / "p-" nudge K1_X by ±1.
// ======================================================
void processTuneCommand(const char *buf);

void Tuning() {
  static char tuneBuf[32];
  static uint8_t tuneIdx = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      tuneBuf[tuneIdx] = '\0';
      if (tuneIdx > 0) processTuneCommand(tuneBuf);
      tuneIdx = 0;
      continue;
    }
    if (tuneIdx < sizeof(tuneBuf) - 1) tuneBuf[tuneIdx++] = c;
  }
}

void processTuneCommand(const char *buf) {
  char param = buf[0];
  const char *rest = buf + 1;
  while (*rest == ' ' || *rest == '=' || *rest == '\t') rest++;

  // Calibration (state command, not a value)
  if (param == 'c') {
    if (buf[1] == '+' && !calibrating) {
      calibrating = true;
      Serial.println(F("Calibrating ON — hold at balance point, then send c-"));
      return;
    }
    if (buf[1] == '-' && calibrating) {
      calibrating = false;
      Serial.print(F("X balance angle: ")); Serial.println(robot_angleX);
      Serial.print(F("Y balance angle: ")); Serial.println(robot_angleY);

      if (abs(robot_angleX) < 30.0f && abs(robot_angleY) < 30.0f) {
        offsets.ID = 33;
        offsets.X  = robot_angleX;
        offsets.Y  = robot_angleY;
        EEPROM.put(0, offsets);

        angleX = 0.0f; angleY = 0.0f;
        angleX_integral = 0.0f; angleY_integral = 0.0f;
        wheelSpeedX_cps = 0.0f; wheelSpeedY_cps = 0.0f;
        k3_pwm_X = 0.0f; k3_pwm_Y = 0.0f;
        wasBalancing = false; k3Scale = 0.0f;

        noInterrupts();
        encoderCountX = 0;
        encoderCountY = 0;
        interrupts();

        calibrated = true;
        Serial.println(F("Balance point saved to EEPROM"));
      } else {
        Serial.println(F("Angle out of range — calibration NOT saved."));
        calibrated = false;
      }
      return;
    }
    return;
  }

  // Relay auto-tune (state command):  t+ / t-  for X,  T+ / T-  for Y
  if (param == 't' || param == 'T') {
    if (buf[1] == '+') {
      if (param == 't') startRelayTuneX();
      else              startRelayTuneY();
    } else if (buf[1] == '-') {
      if (relayTuningX || relayTuningY) Serial.println(F("Relay tune aborted."));
      relayTuningX = false;
      relayTuningY = false;
    }
    return;
  }

  // Print current values
  if (param == '?') { printValues(); return; }

  // Resolve which gain to write
  float *target = nullptr;
  bool clampNonNeg = false;
  switch (param) {
    case 'p': target = &K1_X; break;
    case 'o': target = &KI_X; clampNonNeg = true; break;
    case 'i': target = &K2_X; break;
    case 's': target = &K3_X; clampNonNeg = true; break;
    case 'P': target = &K1_Y; break;
    case 'O': target = &KI_Y; clampNonNeg = true; break;
    case 'I': target = &K2_Y; break;
    case 'S': target = &K3_Y; clampNonNeg = true; break;
    default:
      Serial.print(F("Unknown command: ")); Serial.println(buf);
      return;
  }

  // Increment shortcuts: e.g. "p+" or "p-" (no value after)
  if (buf[1] == '+' && buf[2] == '\0') {
    float step = (param=='o'||param=='O') ? 0.05f :
                 (param=='i'||param=='I') ? 0.1f  :
                 (param=='s'||param=='S') ? 0.001f : 1.0f;
    *target += step;
  } else if (buf[1] == '-' && buf[2] == '\0') {
    float step = (param=='o'||param=='O') ? 0.05f :
                 (param=='i'||param=='I') ? 0.1f  :
                 (param=='s'||param=='S') ? 0.001f : 1.0f;
    *target -= step;
  } else {
    // Direct value entry
    if (*rest == '\0') {
      Serial.println(F("Missing value. Example: p 90.5"));
      return;
    }
    *target = atof(rest);
  }

  if (clampNonNeg && *target < 0) *target = 0;
  printValues();
}

// ======================================================
// Print gains
// ======================================================
void printValues() {
  Serial.print(F("K1_X:")); Serial.print(K1_X);
  Serial.print(F(" KI_X:")); Serial.print(KI_X, 3);
  Serial.print(F(" K2_X:")); Serial.print(K2_X, 3);
  Serial.print(F(" K3_X:")); Serial.print(K3_X, 4);
  Serial.print(F(" | K1_Y:")); Serial.print(K1_Y);
  Serial.print(F(" KI_Y:")); Serial.print(KI_Y, 3);
  Serial.print(F(" K2_Y:")); Serial.print(K2_Y, 3);
  Serial.print(F(" K3_Y:")); Serial.println(K3_Y, 4);
}

// ======================================================
// Print debug
// ======================================================
void printDebug() {
  noInterrupts();
  long countX = encoderCountX;
  long countY = encoderCountY;
  interrupts();

  Serial.print(F("AX:")); Serial.print(angleX, 2);
  Serial.print(F(" AY:")); Serial.print(angleY, 2);
  Serial.print(F(" | IX:")); Serial.print(angleX_integral, 2);
  Serial.print(F(" IY:")); Serial.print(angleY_integral, 2);
  Serial.print(F(" | GZ:")); Serial.print(gyroZfilt, 1);
  Serial.print(F(" GY:")); Serial.print(gyroYfilt, 1);
  Serial.print(F(" | PWM_X:")); Serial.print(pwm_X);
  Serial.print(F(" PWM_Y:")); Serial.print(pwm_Y);
  Serial.print(F(" | K3X:")); Serial.print(k3_pwm_X, 1);
  Serial.print(F(" K3Y:")); Serial.print(k3_pwm_Y, 1);
  Serial.print(F(" | encX:")); Serial.print(countX);
  Serial.print(F(" encY:")); Serial.print(countY);
  Serial.print(F(" | spdX:")); Serial.print(wheelSpeedX_cps, 0);
  Serial.print(F(" spdY:")); Serial.print(wheelSpeedY_cps, 0);
  Serial.print(F(" | vert:")); Serial.print(vertical ? F("1") : F("0"));
  Serial.print(F(" cal:")); Serial.println(calibrated ? F("1") : F("0"));
}
