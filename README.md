# Smarter Page Turner

**Intelligentes System zum automatischen Umblättern von Notenblättern**

Entwickelt im Rahmen der Studienarbeit an der Hochschule Offenburg (2026)

---

## 🎵 Überblick

Der Smarter Page Turner ist ein Echtzeit-System, das die gespielte Musik eines Musikers analysiert und automatisch die Notenblätter auf einem Tablet umblättert.

**Funktionsweise:**
1. 🎤 Mikrofon nimmt Live-Musik auf
2. 📊 Chroma-Features werden in Echtzeit extrahiert
3. 🎯 Online-DTW-Algorithmus vergleicht mit Referenz-Partitur
4. 📍 Aktuelle Position wird getrackt
5. 📱 Bei bestimmter Position → Bluetooth-Signal zum Umblättern

---

## 📁 Projekt-Struktur

```
2.-Studienarbeit-Smarter-Page-Turner/
│
├── Smarter Page Turner/         # Teensy 4.1 Firmware (Hauptsystem)
│   ├── src/                     #   - Audio-Erfassung (I2S)
│   ├── lib/                     #   - Chroma-Extraktion (FFT)
│   │   ├── AudioDSP/            #   - Online-DTW-Tracking
│   │   └── ODTW/
│   └── README.md
│
├── Bluetooth-Manager/           # ESP32-C3 Firmware (BLE Keyboard)
│   ├── src/                     #   - Empfängt Befehle vom Teensy
│   └── README.md                #   - Sendet via Bluetooth an Tablet
│
├── Offline Programme/           # Python-Tools für Datenverarbeitung
│   ├── MusescoreToChroma/       #   - MusicXML → Chroma Konverter
│   ├── PdfToChroma/             #   - PDF → Chroma (via OMR)
│   ├── DTW_Studies/             #   - DTW-Algorithmus Tests
│   │   ├── ODTW_Python/         #   - Python-Prototyp & Parameter-Tuning
│   │   └── DTW-Simulationen.ipynb
│   ├── data/                    #   - Zentrale Daten
│   │   ├── soundfonts/          #   - MuseScore_General.sf2
│   │   ├── audio/               #   - .wav Dateien
│   │   ├── inputs/              #   - .musicxml Partituren
│   │   └── generated/           #   - ScoreData.h Outputs
│   └── README.md
│
└── README.md                    # Diese Datei
```

---

## 🚀 Quick Start

### 1. **Hardware aufbauen**

**Benötigt:**
- Teensy 4.1
- I2S Mikrofon (SPH0645 oder INMP441)
- ESP32-C3 DevKit
- Tablet/iPad mit Noten-App

**Verkabelung:**
- Teensy Pin 1 → ESP32 GPIO 20
- Teensy Pin 0 → ESP32 GPIO 21
- I2S Mikrofon → Teensy (siehe [Smarter Page Turner/README.md](Smarter%20Page%20Turner/README.md))

### 2. **Partitur vorbereiten**

```bash
cd "Offline Programme/MusescoreToChroma"
python musescore_to_chroma.py ../data/inputs/meine_partitur.musicxml --bpm 80 --instrument violin
# → Generiert ScoreData.h
cp ScoreData.h "../../Smarter Page Turner/lib/ODTW/"
```

### 3. **Firmware flashen**

**Teensy:**
```bash
cd "Smarter Page Turner"
pio run -e odtw --target upload
```

**ESP32:**
```bash
cd "Bluetooth-Manager"
pio run --target upload
```

### 4. **Tablet verbinden**

1. Bluetooth aktivieren
2. Mit `"Teensy-PageTurner"` verbinden
3. Noten-App öffnen (z.B. forScore, MobileSheets)
4. Spielen und automatisch umblättern lassen! 🎹

---

## 🛠️ Technologie-Stack

### Embedded (C++)
- **Teensy 4.1** (600 MHz ARM Cortex-M7)
  - Teensy Audio Library (FFT)
  - Custom Online-DTW Implementierung
- **ESP32-C3** (160 MHz RISC-V)
  - NimBLE-Arduino (BLE Stack)
  - ESP32-BLE-Keyboard Library

### Offline Tools (Python)
- **librosa** - Audio-Analyse & Chroma-Extraktion
- **music21** - MusicXML Parsing
- **numpy/scipy** - Numerische Berechnungen
- **matplotlib** - Visualisierung
- **oemer** - Optical Music Recognition (PDF → MusicXML)
- **fluidsynth** - MIDI → Audio Synthese

### Algorithmus
- **Chroma Features** (12 Bins: C, C#, D, ..., B)
- **Online Dynamic Time Warping (ODTW)**
  - Cosine Distance
  - Damping Factor: 0.96
  - Penalties: Wait (0.4), Skip (0.1)
  - Search Window: ±100 Frames

---

## 📖 Dokumentation

Jeder Unterordner hat eine eigene README mit Details:

- **[Smarter Page Turner/](Smarter%20Page%20Turner/README.md)** - Teensy Firmware, Pin-Belegung, Parameter
- **[Bluetooth-Manager/](Bluetooth-Manager/README.md)** - ESP32 Firmware, BLE-Protokoll
- **[Offline Programme/](Offline%20Programme/README.md)** - Python-Tools, Workflow
- **[ODTW_Python/](Offline%20Programme/DTW_Studies/ODTW_Python/README.md)** - DTW-Algorithmus Details

---

## 🧪 Testing & Development

### Python-Prototyp (Recommended)
```bash
cd "Offline Programme/DTW_Studies/ODTW_Python"
python test_robustness.py  # Robustheitstests
python dtw_engine.py        # Live-Test mit Mikrofon
```

### Teensy Debugging
```bash
cd "Smarter Page Turner"
pio run -e mic_test --target upload  # Audio-Test
pio run -e odtw --target upload && pio device monitor  # Tracking-Test
```

### Parameter-Tuning
1. Python-Prototyp für Simulationen nutzen
2. Optimale Parameter finden
3. In `Smarter Page Turner/lib/ODTW/Settings.h` übertragen

---

## 📦 Dependencies

### Python
```bash
pip install numpy librosa sounddevice scipy matplotlib music21 oemer fluidsynth
brew install fluidsynth  # macOS
```

### PlatformIO (automatisch installiert)
```bash
pip install platformio
```

---

## 🎯 Features

✅ **Echtzeit-Tracking** - < 15ms Latenz
✅ **Tempo-Robust** - Funktioniert bei ±20% Tempo-Änderungen
✅ **Noise-Tolerant** - Funktioniert auch bei schlechter Aufnahmequalität
✅ **Wireless** - Bluetooth Low Energy
✅ **Universal** - Funktioniert mit den meisten Noten-Apps
✅ **Open Source** - Komplett quelloffen

---

## 🔧 Hardware

**Empfohlene Komponenten:**
- Teensy 4.1 - ~30€
- ESP32-C3 DevKit - ~5€
- I2S Mikrofon (SPH0645) - ~8€
- Breadboard + Kabel - ~10€

**Gesamt: ~50€**

---

## 📚 Wissenschaftlicher Hintergrund

**Dynamic Time Warping (DTW):**
- Sakoe & Chiba (1978): "Dynamic programming algorithm optimization for spoken word recognition"
- Online-Variante für Echtzeit-Anwendungen

**Chroma Features:**
- Müller & Ewert (2011): "Chroma Toolbox: MATLAB implementations for extracting variants of chroma-based audio features"

---

## 👤 Author

**Samuel Geffert**
- Hochschule Offenburg
- Studienarbeit 2026

---

## 📄 Lizenz

Dieses Projekt ist Teil einer Studienarbeit und dient ausschließlich zu Forschungszwecken.

---

## 🙏 Danksagungen

- Teensy Community für die exzellente Audio Library
- T-vK für die ESP32-BLE-Keyboard Library
- Music21 Team für das großartige MusicXML-Framework

---

**Status:** ✅ Prototyp funktional | 🚧 In Entwicklung
**Letztes Update:** Februar 2026