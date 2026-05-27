// ======================================================
// imu_motors.ino — hardware drivers
//   - IMU init & angle calculation
//   - Motor control
//   - Encoder setup, ISRs, speed estimation
// All globals are defined in main.ino
// ======================================================

// ======================================================
// writeTo
// ======================================================
void writeTo(byte device, byte address, byte value) {
  Wire.beginTransmission(device);
  Wire.write(address);
  Wire.write(value);
  Wire.endTransmission(true);
}

// ======================================================
// Encoder X ISR: A = D2, B = D3
// ======================================================
void encoderX_ISR() {
  int A = digitalRead(ENC_X_A);
  int B = digitalRead(ENC_X_B);
  if (A == B) encoderCountX--;
  else        encoderCountX++;
}

void encoderX_ISR_B() {
  int A = digitalRead(ENC_X_A);
  int B = digitalRead(ENC_X_B);
  if (A == B) encoderCountX++;
  else        encoderCountX--;
}

// ======================================================
// Encoder Y ISRs: A = D7, B = D8
// (Teensy 4.0 — all pins support attachInterrupt)
// ======================================================
void encoderY_ISR_A() {
  int A = digitalRead(ENC_Y_A);
  int B = digitalRead(ENC_Y_B);
  if (A == B) encoderCountY--;
  else        encoderCountY++;
}

void encoderY_ISR_B() {
  int A = digitalRead(ENC_Y_A);
  int B = digitalRead(ENC_Y_B);
  if (A == B) encoderCountY++;
  else        encoderCountY--;
}

// ======================================================
// Setup encoders
// ======================================================
void setupEncoders() {
  pinMode(ENC_X_A, INPUT_PULLUP);
  pinMode(ENC_X_B, INPUT_PULLUP);
  pinMode(ENC_Y_A, INPUT_PULLUP);
  pinMode(ENC_Y_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_X_A), encoderX_ISR,   CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_X_B), encoderX_ISR_B, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_Y_A), encoderY_ISR_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_Y_B), encoderY_ISR_B, CHANGE);
}

// ======================================================
// Encoder speed estimation (called every 10 ms)
// ======================================================
void updateEncoderSpeed() {
  unsigned long now_us = micros();
  if (now_us - lastSpeedTime_us >= SPEED_UPDATE_US) {
    noInterrupts();
    long countX = encoderCountX;
    long countY = encoderCountY;
    interrupts();

    long deltaX = countX - lastEncoderCountX;
    long deltaY = countY - lastEncoderCountY;
    float dt = (now_us - lastSpeedTime_us) * 1e-6f;

    if (dt > 0.0f) {
      wheelSpeedX_cps = SPEED_ALPHA * (deltaX / dt) + (1.0f - SPEED_ALPHA) * wheelSpeedX_cps;
      wheelSpeedY_cps = SPEED_ALPHA * (deltaY / dt) + (1.0f - SPEED_ALPHA) * wheelSpeedY_cps;
    }

    lastEncoderCountX = countX;
    lastEncoderCountY = countY;
    lastSpeedTime_us  = now_us;
  }
}

// ======================================================
// IMU setup — gyro + accel calibration, filter settle
// Hold robot vertical and still during startup
// ======================================================
void angle_setup() {
  Wire.begin();
  Wire.setClock(400000);
  delay(100);

  writeTo(MPU6050, PWR_MGMT_1,   0);
  writeTo(MPU6050, 0x1A,         4);  // DLPF: 21 Hz accel / 20 Hz gyro
  writeTo(MPU6050, ACCEL_CONFIG, accSens  << 3);
  writeTo(MPU6050, GYRO_CONFIG,  gyroSens << 3);
  delay(100);

  // ---- Gyro offset calibration ----
  Serial.println(F("Calibrating gyro..."));
  GyY_offset_sum = 0;
  GyZ_offset_sum = 0;

  for (int i = 0; i < 1024; i++) {
    Wire.beginTransmission(MPU6050);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU6050, (uint8_t)6, (bool)true);
    Wire.read(); Wire.read();  // skip GyX
    int32_t gy = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
    int32_t gz = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
    GyY_offset_sum += gy;
    GyZ_offset_sum += gz;
    delay(5);
  }
  GyY_offset = GyY_offset_sum >> 10;
  GyZ_offset = GyZ_offset_sum >> 10;
  Serial.print(F("GyY offset = ")); Serial.println(GyY_offset);
  Serial.print(F("GyZ offset = ")); Serial.println(GyZ_offset);

  // ---- Accelerometer offset calibration ----
  // At vertical: AcX ~ -16384 (1g), AcY ~ 0, AcZ ~ 0
  Serial.println(F("Calibrating accelerometer..."));
  int32_t AcX_sum = 0, AcY_sum = 0, AcZ_sum = 0;

  for (int i = 0; i < 512; i++) {
    Wire.beginTransmission(MPU6050);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU6050, (uint8_t)6, (bool)true);
    AcX_sum += (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
    AcY_sum += (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
    AcZ_sum += (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
    delay(5);
  }

  int16_t AcX_mean = AcX_sum >> 9;
  int16_t AcY_mean = AcY_sum >> 9;
  int16_t AcZ_mean = AcZ_sum >> 9;

  AcX_offset = -16384 - AcX_mean;
  AcY_offset = -AcY_mean;
  AcZ_offset = -AcZ_mean;

  Serial.print(F("AcX_offset = ")); Serial.println(AcX_offset);
  Serial.print(F("AcY_offset = ")); Serial.println(AcY_offset);
  Serial.print(F("AcZ_offset = ")); Serial.println(AcZ_offset);

  // ---- Reset and settle complementary filter ----
  robot_angleX = 0.0f;
  robot_angleY = 0.0f;

  Serial.println(F("Settling angle filter..."));
  for (int i = 0; i < 200; i++) {
    angle_calc();
    delay(5);
  }

  Serial.print(F("Settled angleX = ")); Serial.println(robot_angleX);
  Serial.print(F("Settled angleY = ")); Serial.println(robot_angleY);
}

// ======================================================
// IMU angle calculation (called every 1 ms)
// ======================================================
void angle_calc() {
  Wire.beginTransmission(MPU6050);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU6050, (uint8_t)14, (bool)true);
  AcX = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
  AcY = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
  AcZ = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
  Wire.read(); Wire.read();  // skip temp
  Wire.read(); Wire.read();  // skip GyX
  GyY = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());
  GyZ = (int16_t)((unsigned)Wire.read() << 8 | Wire.read());

  AcX += AcX_offset;
  AcY += AcY_offset;
  AcZ += AcZ_offset;
  GyY -= GyY_offset;
  GyZ -= GyZ_offset;

  robot_angleX += GyZ * loop_time_s / 65.536f;
  Acc_angleX    = atan2(AcY, -AcX) * 57.2958f;
  robot_angleX  = robot_angleX * Gyro_amount + Acc_angleX * (1.0f - Gyro_amount);
  angleX        = robot_angleX - offsets.X;

  robot_angleY += GyY * loop_time_s / 65.536f;
  Acc_angleY    = -atan2(AcZ, -AcX) * 57.2958f;
  robot_angleY  = robot_angleY * Gyro_amount + Acc_angleY * (1.0f - Gyro_amount);
  angleY        = robot_angleY - offsets.Y;

  if (abs(angleX) > 10.0f || abs(angleY) > 10.0f) vertical = false;
  if (abs(angleX) < 1.0f  && abs(angleY) < 1.0f  &&
      abs(gyroZfilt) < 30.0f && abs(gyroYfilt) < 30.0f) vertical = true;
}

// ======================================================
// Motor control (255 = stop, 0 = full — Nidec 24H convention)
//
// - MOTOR_DEADBAND: |pwm| at or below this -> stop, don't toggle DIRECTION.
//   Kills noise-driven direction chatter near zero.
// - DIR_DEADTIME_US: dead time inserted before a DIRECTION flip, with PWM
//   forced to stop first. Prevents the internal driver from being told to
//   reverse while windings still carry torque (the "click" / shoot-through).
// ======================================================
static const int MOTOR_DEADBAND   = 2;
static const int DIR_DEADTIME_US  = 60;

void Motor_controlX(int pwm) {
  static int8_t lastDirX = 1;

  if (abs(pwm) <= MOTOR_DEADBAND) {
    analogWrite(PWM_X, 255);
    return;
  }

  int8_t newDir = (pwm < 0) ? -1 : 1;
  if (newDir != lastDirX) {
    analogWrite(PWM_X, 255);
    delayMicroseconds(DIR_DEADTIME_US);
    digitalWrite(DIRECTION_X, newDir > 0 ? HIGH : LOW);
    lastDirX = newDir;
  }

  int mag = (pwm < 0) ? -pwm : pwm;
  analogWrite(PWM_X, 255 - constrain(mag, 0, MAX_PWM_CMD));
}

void Motor_controlY(int pwm) {
  static int8_t lastDirY = 1;

  if (abs(pwm) <= MOTOR_DEADBAND) {
    analogWrite(PWM_Y, 255);
    return;
  }

  int8_t newDir = (pwm < 0) ? -1 : 1;
  if (newDir != lastDirY) {
    analogWrite(PWM_Y, 255);
    delayMicroseconds(DIR_DEADTIME_US);
    digitalWrite(DIRECTION_Y, newDir > 0 ? HIGH : LOW);
    lastDirY = newDir;
  }

  int mag = (pwm < 0) ? -pwm : pwm;
  analogWrite(PWM_Y, 255 - constrain(mag, 0, MAX_PWM_CMD));
}
