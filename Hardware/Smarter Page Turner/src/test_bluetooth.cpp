#include <Arduino.h>

// Auf dem Teensy 4.1
void setup() {
  // Serial1 sind Pins 0(RX1) und 1(TX1) am Teensy 4.1
  Serial1.begin(115200);
}

void loop() {
    Serial1.print('n'); // Sende 'n' an den ESP32
    delay(2000); // Entprellen
}