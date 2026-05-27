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

const int MAX_PWM_CMD   = 150;
const int K3_PWM_LIMIT  = 60;

const unsigned long SPEED_UPDATE_US = 10000;
const float SPEED_ALPHA = 0.35;

const unsigned long K3_DELAY_MS = 200;
const unsigned long K3_RAMP_MS  = 400;

bool wasBalancing = false;
unsigned long balanceStartTime = 0;

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

const unsigned long LOOP_PERIOD_US = 1000UL;
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

int32_t GyY, GyZ;

#define accSens     0
#define gyroSens    1
#define Gyro_amount 0.996

int16_t AcX_offset = 0;
int16_t AcY_offset = 0;
int16_t AcZ_offset = 0;

int16_t GyY_offset = 0;
int16_t GyZ_offset = 0;

float robot_angleX = 0.0;
float robot_angleY = 0.0;
float angleX = 0.0;
float angleY = 0.0;

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
  digitalWrite(DIRECTION_X, HIGH);
  digitalWrite(DIRECTION_Y, HIGH);

  setupEncoders();
  delay(200);

  EEPROM.get(0, offsets);
  if (offsets.ID == 33) {
    calibrated = true;
    Serial.println(F("EEPROM offsets loaded"));
  } else {
    offsets.ID = 0;
    offsets.X  = 0.0f;
    offsets.Y  = 0.0f;
    Serial.println(F("No calibration — send c+ then c-"));
  }

  angle_setup();

  lastSpeedTime_us = micros();
  previousT_1_us   = micros();
  previousT_2_us   = micros();
}

// ======================================================
// Main loop
// ======================================================
void loop() {
  currentT_us = micros();

  // 1 ms control loop
  if (currentT_us - previousT_1_us >= LOOP_PERIOD_US) {

    loop_time_s = (currentT_us - previousT_1_us) * 1e-6f;

    Tuning();
    updateEncoderSpeed();

    angle_calc();

    gyroZfilt = alpha * (GyZ / 65.536f) + (1.0f - alpha) * gyroZfilt;
    gyroYfilt = alpha * (GyY / 65.536f) + (1.0f - alpha) * gyroYfilt;

    if (vertical && calibrated) {
      if (!wasBalancing) {
        balanceStartTime = millis();
        wasBalancing = true;
      }

      float k3Scale = getK3Scale();

      angleX_integral += angleX * loop_time_s;
      angleY_integral += angleY * loop_time_s;
      angleX_integral = constrain(angleX_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);
      angleY_integral = constrain(angleY_integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

      k3_pwm_X = constrain(-k3Scale * K3_X * wheelSpeedX_cps, -K3_PWM_LIMIT, K3_PWM_LIMIT);
      k3_pwm_Y = constrain(-k3Scale * K3_Y * wheelSpeedY_cps, -K3_PWM_LIMIT, K3_PWM_LIMIT);

      pwm_X = constrain(
        K1_X * angleX + KI_X * angleX_integral + K2_X * gyroZfilt + k3_pwm_X,
        -MAX_PWM_CMD, MAX_PWM_CMD);

      pwm_Y = constrain(
        K1_Y * angleY + KI_Y * angleY_integral + K2_Y * gyroYfilt + k3_pwm_Y,
        -MAX_PWM_CMD, MAX_PWM_CMD);

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
    }

    previousT_1_us = currentT_us;

    // 500 ms debug print — runs inside the tick so jitter is absorbed into loop_time_s
    if (currentT_us - previousT_2_us >= 500000UL) {
      printDebug();
      previousT_2_us = currentT_us;
    }
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

  if (param == 'c') {
    if (buf[1] == '+' && !calibrating) {
      calibrating = true;
      Serial.println(F("Calibrating — hold at balance point, send c-"));
      return;
    }
    if (buf[1] == '-' && calibrating) {
      calibrating = false;
      if (abs(robot_angleX) < 30.0f && abs(robot_angleY) < 30.0f) {
        offsets.ID = 33;
        offsets.X  = robot_angleX;
        offsets.Y  = robot_angleY;
        EEPROM.put(0, offsets);

        angleX = 0.0f; angleY = 0.0f;
        angleX_integral = 0.0f; angleY_integral = 0.0f;
        wheelSpeedX_cps = 0.0f; wheelSpeedY_cps = 0.0f;
        k3_pwm_X = 0.0f; k3_pwm_Y = 0.0f;
        wasBalancing = false;

        noInterrupts();
        encoderCountX = 0;
        encoderCountY = 0;
        interrupts();

        calibrated = true;
        Serial.print(F("Saved — X:")); Serial.print(offsets.X, 2);
        Serial.print(F(" Y:"));        Serial.println(offsets.Y, 2);
      } else {
        Serial.println(F("Angle out of range — not saved"));
        calibrated = false;
      }
      return;
    }
    return;
  }

  if (param == '?') { printValues(); return; }

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
    default: return;
  }

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
    if (*rest == '\0') return;
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
// Print debug — angle, gyro, PWM, status
// ======================================================
void printDebug() {
  Serial.print(F("AX:")); Serial.print(angleX, 2);
  Serial.print(F(" AY:")); Serial.print(angleY, 2);
  Serial.print(F(" GZ:")); Serial.print(gyroZfilt, 1);
  Serial.print(F(" GY:")); Serial.print(gyroYfilt, 1);
  Serial.print(F(" PX:")); Serial.print(pwm_X);
  Serial.print(F(" PY:")); Serial.print(pwm_Y);
  Serial.print(F(" V:")); Serial.print(vertical ? 1 : 0);
  Serial.print(F(" C:")); Serial.println(calibrated ? 1 : 0);
}
