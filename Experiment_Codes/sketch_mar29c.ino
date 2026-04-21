#include <ESP32Servo.h>
#include <SCServo.h>

Servo myServo;
Servo myServo2;
const int SERVO_PIN  = 4;
const int SERVO_PIN2 = 2;

// SC Servo bus 1
#define S_RXD  16
#define S_TXD  17
// SC Servo bus 2
#define S_RXD2 18
#define S_TXD2 19

SMS_STS sc;
SMS_STS sc2;

int angleToPos(float angle) {
  return (angle / 360.0) * 4095;
}

void setup() {
  // USB Serial — talk to PC via PySerial
  Serial.begin(115200);

  // SC Servo bus 1
  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  sc.pSerial = &Serial1;
  delay(1000);
  sc.EnableTorque(1, 1);
  delay(50);

  // SC Servo bus 2 — use Serial2 instead of Serial
  Serial2.begin(1000000, SERIAL_8N1, S_RXD2, S_TXD2);
  sc2.pSerial = &Serial2;
  delay(1000);
  sc2.EnableTorque(1, 1);
  delay(50);

  myServo.attach(SERVO_PIN);
  myServo2.attach(SERVO_PIN2);
}

void loop() {
  // Read a command from PC over USB Serial
  // Protocol: "CMD,value1,value2,value3,value4\n"
  // e.g. "SET,90,180,90,90\n"
  // value1 = myServo angle
  // value2 = myServo2 angle
  // value3 = sc angle (degrees)
  // value4 = sc2 angle (degrees)

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("SET,")) {
      // Parse comma-separated values
      int idx = 4; // skip "SET,"
      float vals[4] = {0, 0, 0, 0};
      int v = 0;
      String token = "";

      for (int i = idx; i <= line.length() && v < 4; i++) {
        char c = (i < line.length()) ? line[i] : ',';
        if (c == ',') {
          vals[v++] = token.toFloat();
          token = "";
        } else {
          token += c;
        }
      }

      // myServo angle
      myServo.write((int)vals[0]);

      // myServo2 angle
      myServo2.write((int)vals[3]);

      // SC Servo 1
      int pos = angleToPos(vals[1]);
      pos = constrain(pos, 0, 4095);
      pos = 4095 - pos;
      sc.WritePosEx(1, pos, 0, 100);

      // SC Servo 2
      int pos2 = angleToPos(vals[2]);
      pos2 = constrain(pos2, 0, 4095);
      pos2 = 4095 - pos2;
      sc2.WritePosEx(1, pos2, 0, 100);

      Serial.println("OK");
    }
  }
}
