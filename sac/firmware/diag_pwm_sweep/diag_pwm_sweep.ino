/*
 * diag_pwm_sweep.ino — measure the Nidec arm's RAW duty -> speed curve.
 *
 * Purpose: the normal firmware floors the motor at MIN_SPEED=0.06, so the arm is
 * "stopped or ~13 rad/s, nothing between". This diagnostic drives RAW PWM duty
 * (no MIN_SPEED mapping) at many low levels and measures the resulting arm speed,
 * so we can see the true low-speed behaviour and design a fine speed map.
 *
 * Method (cable-safe, autonomous): for each duty level, drive the arm back and
 * forth across +-LIMIT_DEG and time each traversal. speed = swept angle / time.
 * Reverses in firmware (no serial latency) and hard-stops at HARD_DEG.
 *
 * Pins (same as nidec_policy.ino):
 *   PWM  D25 (inverted: 0% = full speed, 100% = stop), DIR D26, BRAKE D27 (HIGH=run)
 *   Encoder CHA/CHB on D34/D35, ~200 CPR (A-only CHANGE decode)
 *
 * Use: flash, open serial @ 921600. Hand-center the arm (cable neutral). Send "go".
 *   Prints one line per duty:  duty=0.040 speed=2.35 rad/s (n=3)
 *   or  duty=0.020 STALLED
 * Send "s" any time to abort. Prints "DONE" at the end.
 */
#include <Arduino.h>

#define PWM_PIN   25
#define DIR_PIN   26
#define BRAKE_PIN 27
const int PWM_FREQ = 20000;
const int PWM_RES_BITS = 8;
const int PWM_STOP = 255;            // inverted: 100% duty = stop

#define ENC_A 34
#define ENC_B 35
const float CPR = 200.0;

const float LIMIT_DEG = 55.0;        // reverse here
const float HARD_DEG  = 100.0;       // emergency stop (cable limit ~120)
const unsigned long TRAVERSAL_TIMEOUT_MS = 3000;

// Raw duty levels to test (fraction of full speed, BEFORE any MIN_SPEED mapping).
float duties[] = {0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080,
                  0.100, 0.120, 0.150, 0.200};
const int N_DUTY = sizeof(duties) / sizeof(duties[0]);

volatile long enc_count = 0;

void IRAM_ATTR encISR() {
  int a = digitalRead(ENC_A);
  int b = digitalRead(ENC_B);
  if (a == b) enc_count++; else enc_count--;
}

float phiDeg() {
  noInterrupts();
  long c = enc_count;
  interrupts();
  return c * 360.0 / CPR;
}

void setDrive(float frac, int dir) {
  if (frac <= 0.0) { ledcWrite(PWM_PIN, PWM_STOP); return; }
  if (frac > 1.0) frac = 1.0;
  digitalWrite(BRAKE_PIN, HIGH);
  digitalWrite(DIR_PIN, dir > 0 ? LOW : HIGH);   // +dir -> phi increases (matches nidec_policy)
  int duty = (int)((1.0 - frac) * 255.0);        // inverted PWM
  if (duty < 0) duty = 0;
  if (duty > 255) duty = 255;
  ledcWrite(PWM_PIN, duty);
}

void motorStop() {
  ledcWrite(PWM_PIN, PWM_STOP);
  digitalWrite(BRAKE_PIN, LOW);
}

bool abortRequested() {
  if (Serial.available()) {
    String c = Serial.readStringUntil('\n');
    c.trim();
    if (c == "s") return true;
  }
  return false;
}

// Measure mean traversal speed [rad/s] at a duty. Returns 0 if stalled, -1 if aborted.
float measureSpeed(float frac) {
  const float limit = LIMIT_DEG;
  float sum = 0.0; int n = 0;
  int dir = (phiDeg() > 0) ? -1 : +1;            // head toward the side with room
  for (int t = 0; t < 4 && n < 3; t++) {
    float startPhi = phiDeg();
    float target = (dir > 0) ? +limit : -limit;
    unsigned long t0 = millis();
    bool reached = false;
    setDrive(frac, dir);
    while (millis() - t0 < TRAVERSAL_TIMEOUT_MS) {
      float p = phiDeg();
      if (fabs(p) > HARD_DEG) { motorStop(); return -1; }
      if ((dir > 0 && p >= target) || (dir < 0 && p <= target)) { reached = true; break; }
      if (abortRequested()) { motorStop(); return -1; }
      delay(2);
    }
    unsigned long dt = millis() - t0;
    float endPhi = phiDeg();
    setDrive(0.0, dir);                           // brake between traversals
    delay(200);
    if (reached && dt > 0) {
      float dist_rad = fabs(endPhi - startPhi) * PI / 180.0;
      sum += dist_rad / (dt / 1000.0);
      n++;
    }
    dir = -dir;
  }
  if (n == 0) return 0.0;
  return sum / n;
}

void runSweep() {
  Serial.println("# duty sweep start (hand-center first). speed = swept angle / time.");
  for (int i = 0; i < N_DUTY; i++) {
    float frac = duties[i];
    float spd = measureSpeed(frac);
    if (spd < 0) { Serial.println("# ABORTED"); break; }
    if (spd == 0.0) {
      Serial.print("duty="); Serial.print(frac, 3); Serial.println(" STALLED");
    } else {
      Serial.print("duty="); Serial.print(frac, 3);
      Serial.print(" speed="); Serial.print(spd, 3); Serial.println(" rad/s");
    }
  }
  motorStop();
  Serial.println("DONE");
}

void setup() {
  Serial.begin(921600);
  delay(300);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(BRAKE_PIN, OUTPUT);
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_RES_BITS);
  motorStop();
  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encISR, CHANGE);

  Serial.println("# diag_pwm_sweep ready. Hand-center the arm (cable neutral).");
  Serial.println("# Send 'go' to start the raw-duty sweep, 's' to abort.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "go") {
      runSweep();
    } else if (cmd == "s") {
      motorStop();
    }
  }
}
