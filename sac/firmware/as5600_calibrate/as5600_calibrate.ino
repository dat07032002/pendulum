/*
 * AS5600 calibration sketch for Furuta pendulum.
 *
 * Flash this to the ESP32, open Serial Monitor at 921600 baud.
 * Let the pendulum hang freely and note the raw value.
 * Then run:  python sac/calibrate_as5600.py --port COM5
 * It will compute the correct UPRIGHT_RAW to put in nidec_policy.ino.
 *
 * Pins: SDA=21, SCL=22  (same as main firmware)
 */

#include <Wire.h>

#define SDA_PIN     21
#define SCL_PIN     22
#define AS5600_ADDR 0x36
#define RAW_ANGLE_H 0x0C
#define AS5600_STATUS 0x0B

uint16_t readRaw() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(RAW_ANGLE_H);
  if (Wire.endTransmission(false) != 0) return 0xFFFF;
  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() < 2) return 0xFFFF;
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return ((hi & 0x0F) << 8) | lo;
}

uint8_t readStatus() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(AS5600_STATUS);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom(AS5600_ADDR, 1);
  return Wire.available() ? Wire.read() : 0;
}

void setup() {
  Serial.begin(921600);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  delay(300);
  Serial.println("# AS5600 calibration — let pendulum hang freely.");
  Serial.println("# Format: raw=<0-4095>  deg=<0-360>  magnet=<OK/WEAK/STRONG>");
}

void loop() {
  uint16_t raw = readRaw();
  uint8_t  st  = readStatus();

  if (raw == 0xFFFF) {
    Serial.println("# ERROR: AS5600 not responding (check I2C wiring)");
    delay(500);
    return;
  }

  float deg = raw * 360.0f / 4096.0f;
  bool  md  = st & 0x20;  // magnet detected
  bool  ml  = st & 0x10;  // magnet too weak
  bool  mh  = st & 0x08;  // magnet too strong

  const char* mag = md ? (ml ? "WEAK" : (mh ? "STRONG" : "OK")) : "NOT_DETECTED";

  Serial.print("raw="); Serial.print(raw);
  Serial.print("  deg="); Serial.print(deg, 2);
  Serial.print("  magnet="); Serial.println(mag);
  delay(200);
}
