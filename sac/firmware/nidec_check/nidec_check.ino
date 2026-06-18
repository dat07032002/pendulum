/*
 * Nidec bring-up CHECK sketch — sensors only, motor NOT driven.
 *
 * RUN THIS WITH THE 24V MOTOR SUPPLY OFF.
 * The encoder + ESP32 are powered by the 5V rail, so they work with 24V off,
 * and with no 24V the motor cannot spin. Totally safe.
 *
 * Purpose:
 *   1. Measure the Nidec encoder CPR  -> turn the ARM by hand exactly 1 rev,
 *      read the enc_count delta.
 *   2. Verify CHA/CHB read (A/B toggle as you turn) through the level shifter.
 *   3. Verify the AS5600 pendulum sensor (theta_deg changes as you move it).
 *
 * Motor pins are set to a safe "stopped" state for when 24V is applied later:
 *   PWM=255 (100% duty = stop, inverted), BRAKE=LOW (not run-enabled), DIR=LOW.
 */

#include <Wire.h>

// --- AS5600 (pendulum theta) ---
#define AS5600_ADDR 0x36
#define RAW_ANGLE_H 0x0C
#define SDA_PIN 21
#define SCL_PIN 22

// --- Nidec motor control pins (NOT driven in this sketch) ---
#define PWM_PIN   25   // inverted: 0% = full, 100% = stop
#define DIR_PIN   26   // CW/CCW
#define BRAKE_PIN 27   // HIGH = run-enabled; LOW = not running (safe here)

// --- Nidec quadrature encoder (phi), 5V -> level shifter -> 3.3V ---
#define ENC_A 34   // CHA  (input-only pin, level-shifted)
#define ENC_B 35   // CHB

const int PWM_FREQ = 20000;
const int PWM_RES_BITS = 8;
const int PWM_STOP = 255;       // 100% duty = stop (inverted PWM)

volatile long enc_count = 0;

void IRAM_ATTR handleEncoder() {
  int a = digitalRead(ENC_A);
  int b = digitalRead(ENC_B);
  if (a == b) enc_count++;
  else        enc_count--;
}

uint16_t readAS5600Raw() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(RAW_ANGLE_H);
  Wire.endTransmission(false);
  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() < 2) return 0;
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return ((hi & 0x0F) << 8) | lo;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  // Hold the motor in a safe stopped state (matters once 24V is on).
  pinMode(DIR_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_RES_BITS);
  ledcWrite(PWM_PIN, PWM_STOP);   // 100% = stop
  digitalWrite(DIR_PIN, LOW);
  digitalWrite(BRAKE_PIN, LOW);   // not run-enabled

  pinMode(ENC_A, INPUT);          // level shifter drives these; add ext. pull-ups if open-collector
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), handleEncoder, CHANGE);

  Serial.println("# Nidec CHECK sketch — motor NOT driven. KEEP 24V SUPPLY OFF.");
  Serial.println("# Send 'z' to zero enc_count.");
  Serial.println("# Turn the ARM by hand exactly 1 full revolution -> enc_count = CPR.");
  Serial.println("# A/B should toggle as you turn; theta_deg should change as you move the pendulum.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "z") {
      noInterrupts(); enc_count = 0; interrupts();
      Serial.println("# enc_count zeroed");
    }
  }

  noInterrupts();
  long c = enc_count;
  interrupts();

  uint16_t raw = readAS5600Raw();
  float theta_deg = (raw / 4096.0) * 360.0;

  Serial.print("enc_count="); Serial.print(c);
  Serial.print("  A="); Serial.print(digitalRead(ENC_A));
  Serial.print(" B="); Serial.print(digitalRead(ENC_B));
  Serial.print("  AS5600_raw="); Serial.print(raw);
  Serial.print("  theta_deg="); Serial.println(theta_deg, 2);

  delay(50);
}
