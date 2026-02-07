# Bluetooth-Manager - ESP32-C3 BLE Keyboard

Bluetooth Low Energy Keyboard Emulator für den Smarter Page Turner.

## 🎯 Funktion

Der ESP32-C3 empfängt Page-Turn-Befehle vom Teensy 4.1 via UART und sendet sie als Bluetooth-Tastatur-Signale an ein Tablet/iPad.

**Flow:**
```
Teensy 4.1 → UART → ESP32-C3 → BLE → Tablet/iPad
(ODTW Tracking)      (BLE Keyboard)   (Noten-App)
```

## 🔧 Hardware

- **ESP32-C3 DevKit-M1**
- **Verbindung zum Teensy:**
  - ESP32 RX (GPIO 20) ← Teensy TX (Pin 1)
  - ESP32 TX (GPIO 21) → Teensy RX (Pin 0)
  - GND ← → GND

## 📡 BLE Keyboard

Der ESP32 meldet sich als:
- **Name**: `"Teensy-PageTurner"`
- **Hersteller**: `"ESP32-Maker"`
- **Typ**: Bluetooth Tastatur

### Unterstützte Tasten

| Befehl | Taste | Verwendung |
|--------|-------|------------|
| `'n'` | `PAGE_DOWN` | Nächste Seite |
| `'p'` | `PAGE_UP` | Vorherige Seite |

**Alternative Keys:** `KEY_RIGHT_ARROW` / `KEY_LEFT_ARROW` (in Code auskommentiert)

## 🚀 Build & Upload

### PlatformIO CLI
```bash
pio run --target upload
pio device monitor
```

### VS Code
1. Öffne PlatformIO Extension
2. Klicke "Upload and Monitor"

## 🔗 UART-Protokoll

**Baud Rate:** 115200
**Format:** 8N1 (8 Data Bits, No Parity, 1 Stop Bit)

### Befehle (vom Teensy)
```cpp
'n'  // Next Page → PAGE_DOWN
'p'  // Previous Page → PAGE_UP
```

**Beispiel (Teensy Code):**
```cpp
Serial1.print('n');  // Sendet Page Down Befehl
```

## 📱 Tablet/iPad Pairing

1. ESP32 mit Strom versorgen
2. Auf Tablet: **Einstellungen → Bluetooth**
3. Warte auf `"Teensy-PageTurner"`
4. Verbinden (kein PIN notwendig)

### Kompatible Apps

Getestet mit:
- **forScore** (iOS) ✅
- **MobileSheets** (Android/iOS) ✅
- **PDF-Reader Apps** (meist `PAGE_DOWN`/`PAGE_UP`)

**Tipp:** Manche Apps bevorzugen Pfeiltasten. Ändere in `main.cpp`:
```cpp
// Statt PAGE_DOWN:
bleKeyboard.write(KEY_RIGHT_ARROW);
```

## 🧪 Testing

### 1. Serial Monitor Test
```bash
pio device monitor
```
**Expected Output:**
```
Starte Page Turner...
Warte auf Bluetooth Verbindung...
[nach Verbindung]
Befehl empfangen: n
```

### 2. Manual UART Test
Mit einem USB-Serial-Adapter:
- Sende `'n'` → Tablet sollte umblättern

### 3. End-to-End Test
Mit Teensy verbunden:
- Teensy sendet `Serial1.print('n')`
- ESP32 empfängt und sendet BLE
- Tablet blättert um

## 📚 Libraries

- **ESP32 BLE Keyboard** (T-vK/ESP32-BLE-Keyboard)
  - GitHub: https://github.com/T-vK/ESP32-BLE-Keyboard
- **NimBLE-Arduino** (1.4.1)
  - Leichtgewichtige BLE-Stack Alternative

## ⚙️ Konfiguration

### main.cpp
```cpp
// BLE Name anpassen
BleKeyboard bleKeyboard("Dein-Name", "Hersteller", 100);

// UART Pins (falls andere Hardware)
#define RX_PIN 20  // ESP32 Empfang
#define TX_PIN 21  // ESP32 Sendung

// Baud Rate (muss mit Teensy übereinstimmen!)
TeensySerial.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);
```

## 🐛 Troubleshooting

### "Warte auf Bluetooth Verbindung..." endlos
- Tablet Bluetooth aktiviert?
- `"Teensy-PageTurner"` in der Geräteliste sichtbar?
- Neustart: ESP32 Strom aus/an

### Keine Befehle empfangen
- UART-Kabel korrekt? (RX ↔ TX gekreuzt!)
- Baud Rate 115200 auf beiden Seiten?
- Serial Monitor: `Befehl empfangen: ...` erscheint?

### Tablet blättert nicht um
- App unterstützt `PAGE_DOWN`? (Versuche `KEY_RIGHT_ARROW`)
- BLE-Verbindung aktiv? (Status-LED auf ESP32?)
- Test mit Bluetooth-Tastatur-Tester-App

## 📝 Pinout ESP32-C3 DevKit-M1

```
     [USB-C]
  ┌─────────────┐
  │   ESP32-C3  │
  │             │
  │ 20 (RX) ────┼─→ Teensy TX (Pin 1)
  │ 21 (TX) ────┼─→ Teensy RX (Pin 0)
  │ GND     ────┼─→ GND
  │             │
  └─────────────┘
```

## 🔗 Integration

Siehe: [`Smarter Page Turner/`](../Smarter%20Page%20Turner/README.md) für Teensy-Seite.

---

**Hardware:** ESP32-C3 DevKit-M1
**Author:** Samuel Geffert
**Projekt:** Smarter Page Turner - Studienarbeit 2026
