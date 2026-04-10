#include <Arduino.h>
#include <BleKeyboard.h>

// --- KONFIGURATION ---
// Name, der im Bluetooth-Menü des Tablets erscheint
BleKeyboard bleKeyboard("Teensy-PageTurner", "ESP32-Maker", 100);

// Pin-Definition für UART zum Teensy
// Beim ESP32-C3 sind GPIO 20 (RX) und 21 (TX) oft die Standard-UART Pins.
// Prüfe dein Pinout!
#define RX_PIN 20 
#define TX_PIN 21

// Initialisiere Hardware Serial 1
HardwareSerial TeensySerial(1);

void setup() {
  // Debug Serial über USB
  Serial.begin(9600);
  Serial.println("Starte Page Turner...");

  // Kommunikation zum Teensy
  // Baudrate muss mit dem Teensy übereinstimmen!
  TeensySerial.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);

  // Starte Bluetooth Tastatur
  bleKeyboard.begin();
}

void loop() {
  // Nur Befehle verarbeiten, wenn Bluetooth verbunden ist
  if (bleKeyboard.isConnected()) {
    
    if (TeensySerial.available()) {
      char command = TeensySerial.read();

      // Debug Ausgabe
      Serial.print("Befehl empfangen: ");
      Serial.println(command);

      switch (command) {
        case 'n': // 'n' für Next (Nächste Seite)
          // Die meisten Musik-Apps reagieren auf Pfeil Rechts, Pfeil Runter oder PageDown
          bleKeyboard.write(KEY_PAGE_DOWN);
          // Alternativ: bleKeyboard.write(KEY_RIGHT_ARROW);
          break;

        case 'p': // 'p' für Previous (Vorherige Seite)
          bleKeyboard.write(KEY_PAGE_UP);
          // Alternativ: bleKeyboard.write(KEY_LEFT_ARROW);
          break;
          
        default:
          Serial.println("Unbekannter Befehl");
          break;
      }
    }
  } else {
    // Optional: Eine LED blinken lassen, wenn nicht verbunden
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 3000) {
      Serial.println("Warte auf Bluetooth Verbindung...");
      lastPrint = millis();
    }
  }
}