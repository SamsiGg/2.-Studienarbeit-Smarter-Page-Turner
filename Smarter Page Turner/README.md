# Smarter Page Turner - Teensy 4.1 Firmware

Embedded Firmware für den intelligenten Notenblatt-Umblätter basierend auf Teensy 4.1.

## 🎯 Funktionsweise

1. **Audio-Erfassung**: Mikrofon nimmt Live-Musik auf (I2S)
2. **Chroma-Extraktion**: Echtzeit FFT + Chroma-Berechnung (AudioDSP Library)
3. **Position-Tracking**: Online-DTW vergleicht mit Referenz-Partitur (ODTW Library)
4. **Page Turn Signal**: Bei bestimmter Position → Bluetooth-Signal

## 📁 Struktur

```
Smarter Page Turner/
├── src/
│   ├── odtw_turner.cpp        # Hauptprogramm (ODTW + Audio)
│   ├── test_mic_chroma.cpp    # Test: Mikrofon + Chroma
│   └── test_bluetooth.cpp     # Test: Bluetooth-Kommunikation
├── lib/
│   ├── AudioDSP/              # FFT + Chroma-Berechnung
│   │   ├── Chroma.h
│   │   └── Chroma.cpp
│   └── ODTW/                  # Online-DTW-Algorithmus
│       ├── DTW.h
│       ├── Settings.h
│       └── ScoreData.h        # Referenz-Partitur (Generated)
└── platformio.ini             # Build-Konfiguration
```

## 🔧 Hardware

- **Teensy 4.1** (600 MHz ARM Cortex-M7)
- **I2S Mikrofon** (z.B. SPH0645 oder INMP441)
- **Bluetooth-Modul** (über Serial1 an ESP32-C3)

### Pin-Belegung

**I2S Audio Input:**
- Pin 7: RX (BCLK)
- Pin 8: TX (not used)
- Pin 20: LRCLK (WS)
- Pin 21: IN (Data)

**Serial1 (Bluetooth):**
- Pin 0: RX
- Pin 1: TX

## 🚀 Build & Upload

### PlatformIO CLI

**Test: Mikrofon + Chroma**
```bash
pio run -e mic_test --target upload
pio device monitor
```

**Test: Bluetooth**
```bash
pio run -e blue_test --target upload
```

**Haupt-Programm (ODTW)**
```bash
pio run -e odtw --target upload
pio device monitor
```

### VS Code

1. Öffne PlatformIO Extension
2. Wähle Environment (`mic_test`, `odtw`, oder `blue_test`)
3. Klicke "Upload and Monitor"

## 📊 Libraries

### AudioDSP
Echtzeit-Audio-Verarbeitung:
- **FFT**: 4096 Samples (Teensy Audio Library)
- **Chroma-Extraktion**: 12 Bins (C, C#, D, ..., B)
- **L2-Normalisierung**: Für robuste Erkennung

### ODTW (Online Dynamic Time Warping)
Position-Tracking-Algorithmus:
- **Cosine Distance**: Vergleicht Chroma-Vektoren
- **Search Window**: ±100 Frames Suchradius
- **Damping Factor**: 0.96 für akkumulierte Kosten
- **Penalties**: Wait (0.4), Skip (0.1)

## 📝 Konfiguration

### Settings.h
Parameter für ODTW und Audio:
```cpp
#define FFT_SIZE 4096
#define HOP_LENGTH 512
#define SEARCH_WINDOW 100
#define DAMPING_FACTOR 0.96f
#define WAIT_PENALTY 0.4f
#define SKIP_PENALTY 0.1f
```

### ScoreData.h
Referenz-Partitur (wird vom Python-Tool generiert):
```cpp
const int score_len = 1234;        // Anzahl Frames
const int num_pages = 3;           // Anzahl Seiten
const int page_end_indices[] = {400, 800, 1200};  // Seiten-Ende Frames
const float score_chroma[][12] = { ... };  // Chroma-Vektoren
```

**Generierung:**
```bash
cd "../Offline Programme/MusescoreToChroma"
python musescore_to_chroma.py partitur.musicxml --bpm 40 --instrument violin
# → Kopiere ScoreData.h nach lib/ODTW/
```

## 🧪 Testing

### 1. Mikrofon-Test
Prüft Audio-Input und Chroma-Berechnung:
```bash
pio run -e mic_test --target upload && pio device monitor
```
**Expected Output:**
```
Frame: 42
Chroma: C:0.82 C#:0.12 D:0.05 ...
RMS: 0.15
```

### 2. Bluetooth-Test
Prüft Serial-Kommunikation mit ESP32:
```bash
pio run -e blue_test --target upload
```

### 3. ODTW Full Test
Vollständiges System mit Live-Tracking:
```bash
pio run -e odtw --target upload && pio device monitor
```
**Expected Output:**
```
[2.341s] [===>    ] Pos: 215 | Cost: 3.42
```

## ⚙️ Parameter-Tuning

Optimale Parameter wurden mit Python-Prototyp ermittelt (siehe `Offline Programme/DTW_Studies/ODTW_Python/`).

**Wenn Tracking nicht funktioniert:**
1. `WAIT_PENALTY` erhöhen → Verhindert Stehenbleiben
2. `SEARCH_WINDOW` vergrößern → Mehr Toleranz für Tempo
3. `DAMPING_FACTOR` anpassen → 0.9-0.98 (niedrig = aggress

iver)

## 📚 Dependencies

- **Teensy Audio Library** (Built-in)
- **Teensy I2S** (Built-in)

## 🔗 Zusammenarbeit mit ESP32

Der Teensy kommuniziert via Serial1 (9600 baud) mit dem ESP32-C3:
- **Befehl**: `"PAGE_TURN\n"` → ESP32 sendet Bluetooth-Signal
- **Format**: Text-basiert, newline-terminiert

Siehe: [`Bluetooth-Manager/`](../Bluetooth-Manager/README.md)

---

**Hardware:** Teensy 4.1
**Author:** Samuel Geffert
**Projekt:** Smarter Page Turner - Studienarbeit 2026
