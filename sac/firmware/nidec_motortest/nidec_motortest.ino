/*
 * Nidec MOTOR TEST sketch — careful first drive.
 *
 * SAFETY:
 *   - 12V supply ON for this test (strong motor: 12V instead of 24V).
 *   - Keep the area clear and your HAND ON THE POWER SWITCH.
 *   - Pulses are bounded: auto-stop after PULSE_MS, speed hard-capped at 0.20.
 *   - Starts very gentle (0.05) in 0.02 steps. Send 's' anytime for stop.
 *
 * Commands (serial, 115200):
 *   f  -> forward pulse at test_speed
 *   r  -> reverse pulse at test_speed
 *   +  -> test_speed += 0.05
 *   -  -> test_speed -= 0.05
 *   s  -> immediate stop
 *   z  -> zero encoder count
 *
 * Verify:
 *   - motor spins on f / r (note which physical direction each is)
 *   - enc count moves consistently with motion (defines phi sign)
 *   - low speed = gentle motion (confirms PWM inversion: 0%=full,100%=stop)
 */

#include <Wire.h>

#define AS5600_ADDR 0x36
#define RAW_ANGLE_H 0x0C
#define SDA_PIN 21
#define SCL_PIN 22

#define PWM_PIN   25   // inverted: 0% = full, 100% = stop
#define DIR_PIN   26
#define BRAKE_PIN 27   // HIGH = run-enabled

#define ENC_A 34
#define ENC_B 35

const int PWM_FREQ = 20000;
const int PWM_RES_BITS = 8;
const int PWM_STOP = 255;        // 100% duty = stop
const int UPRIGHT_RAW = 27;      // from hanging = 182.37 deg
const unsigned long PULSE_MS = 300;   // short bounded pulse
const float SPEED_CAP = 0.20;    // hard limit for the test (strong motor @ 12V)

float test_speed = 0.05;         // start very gentle (0..1)
volatile long enc_count = 0;
unsigned long pulse_end = 0;
bool running = false;

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

void motorStop() {
  ledcWrite(PWM_PIN, PWM_STOP);   // stop
  digitalWrite(BRAKE_PIN, LOW);   // not run-enabled
  running = false;
}

void motorPulse(bool forward, float speed) {
  if (speed < 0) speed = 0;
  if (speed > SPEED_CAP) speed = SPEED_CAP;
  digitalWrite(BRAKE_PIN, HIGH);                 // run-enabled
  digitalWrite(DIR_PIN, forward ? HIGH : LOW);
  int duty = (int)((1.0 - speed) * 255.0);       // inverted PWM
  ledcWrite(PWM_PIN, duty);
  running = true;
  pulse_end = millis() + PULSE_MS;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  pinMode(DIR_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_RES_BITS);
  motorStop();

  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), handleEncoder, CHANGE);

  Serial.println("# Nidec MOTOR TEST. 12V ON. Hand on power. Area clear.");
  Serial.println("# f=fwd pulse, r=rev pulse, +/- speed, s=stop, z=zero enc");
  Serial.print("# test_speed="); Serial.println(test_speed, 2);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if      (cmd == "s") { motorStop(); Serial.println("# STOP"); }
    else if (cmd == "f") { motorPulse(true,  test_speed); Serial.println("# fwd pulse"); }
    else if (cmd == "r") { motorPulse(false, test_speed); Serial.println("# rev pulse"); }
    else if (cmd == "+") { test_speed += 0.02; if (test_speed > SPEED_CAP) test_speed = SPEED_CAP; Serial.print("# speed="); Serial.println(test_speed, 2); }
    else if (cmd == "-") { test_speed -= 0.02; if (test_speed < 0.02) test_speed = 0.02; Serial.print("# speed="); Serial.println(test_speed, 2); }
    else if (cmd == "z") { noInterrupts(); enc_count = 0; interrupts(); Serial.println("# enc zeroed"); }
  }

  if (running && millis() >= pulse_end) {
    motorStop();
    Serial.println("# pulse done, stopped");
  }

  noInterrupts();
  long c = enc_count;
  interrupts();

  uint16_t raw = readAS5600Raw();
  float theta = (raw - UPRIGHT_RAW) * 360.0 / 4096.0;
  while (theta > 180) theta -= 360;
  while (theta < -180) theta += 360;
  theta = -theta;   // sign convention (hanging ~ -180)

  Serial.print("enc="); Serial.print(c);
  Serial.print(" theta_deg="); Serial.print(theta, 1);
  Serial.print(" running="); Serial.println(running ? 1 : 0);

  delay(50);
}
