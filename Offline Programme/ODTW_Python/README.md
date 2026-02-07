# ODTW Python - Online Dynamic Time Warping

Dieser Ordner enthält die Python-Implementierung und Tests für den Online-DTW-Algorithmus des "Smarter Page Turner" Projekts.

## 📁 Struktur

```
ODTW_Python/
├── dtw_engine.py           # Kern-Modul mit ODTW-Klassen
├── test_robustness.py      # Robustheitstests mit verschiedenen Szenarien
├── audio_generator.py      # Utility zum Generieren von Audio aus Chroma
├── data/                   # Daten-Ordner
│   ├── ScoreData.h         # Referenz-Partitur (Chroma-Daten als C-Header)
│   └── Fiocco-Live (40bpm)_chroma.npy  # Live-Aufnahme als Chroma-Array
└── README.md               # Diese Datei
```

## 🎯 Module

### `dtw_engine.py`
**Hauptmodul** mit den ODTW-Implementierungen:

- **`StandardODTW`**: Standard-Implementierung für Live-Tracking
  - Vergleicht Live-Chroma-Vektoren mit Referenz-Partitur
  - Verwendet Cosine Distance als Ähnlichkeitsmaß
  - Glättung mit Moving Average
  - Dämpfungsfaktor für akkumulierte Kosten

- **`DebugODTW`**: Erweiterte Version mit Debug-Informationen
  - Gibt zusätzlich lokale Kosten zurück
  - Nützlich für Kosten-Analyse und Parameter-Tuning

- **`load_h_file_chroma()`**: Lädt Chroma-Daten aus C-Header-Dateien

- **`main()`**: Live-Test mit Audio-Input
  - Nimmt Audio vom Mikrofon auf
  - Berechnet Chroma-Features in Echtzeit
  - Trackt Position in der Partitur

**Konfigurierbare Parameter:**
```python
SAMPLE_RATE = 44100      # Audio Sample Rate
BLOCK_SIZE = 4096        # FFT Größe
SEARCH_WINDOW = 100      # Suchradius für ODTW
HOP_LENGTH = 512         # FFT Hop Length
DAMPING_FACTOR = 0.96    # Dämpfung für alte Kosten
WAIT_PENALTY = 0.4       # Strafe fürs Stehenbleiben
SKIP_PENALTY = 0.1       # Strafe fürs Überspringen
SMOOTHING_WINDOW = 15    # Moving Average Fenster
```

### `test_robustness.py`
**Robustheitstests** für den ODTW-Algorithmus:

- Unterstützt zwei Input-Formate:
  - `.wav`: Echte Audio-Dateien
  - `.npy`: Vorberechnete Chroma-Daten

- Testet verschiedene Szenarien:
  - **Tempo-Variationen**: 0.8x, 1.0x, 1.2x, 1.3x
  - **Audio-Rauschen**: Simuliert schlechte Aufnahmebedingungen
  - **Kombinationen**: Schnell + Rauschen, Langsam + Rauschen

- Zwei Analyse-Modi:
  1. **Tracking Comparison**: Einfacher Vergleich der Tracking-Pfade
  2. **Kosten-Analyse**: Detaillierte Analyse der akkumulierten Kosten

**Verwendung:**
```bash
python test_robustness.py
# Interaktive Auswahl von Input-Typ und Analyse-Modus
```

### `audio_generator.py`
**Utility-Tool** zum Generieren von Audio aus Chroma-Vektoren:

- Lädt Chroma-Daten aus `.h` oder `.npy` Dateien
- Findet dominanten Ton pro Frame
- Generiert Sinus-Ton-Melodie
- Nützlich zum Verifizieren von Chroma-Daten

**Verwendung:**
```bash
python audio_generator.py data/ScoreData.h --out melody.wav
python audio_generator.py data/Fiocco-Live\ \(40bpm\)_chroma.npy --out live_melody.wav
```

## 🚀 Schnellstart

### Live-Test mit Audio-Input
```bash
python dtw_engine.py
```
→ Startet Live-Tracking mit Mikrofon-Input

### Robustheitstests
```bash
python test_robustness.py
```
→ Interaktive Tests mit verschiedenen Szenarien

### Audio aus Chroma generieren
```bash
python audio_generator.py data/ScoreData.h
```
→ Erstellt `dominant_tone_melody.wav`

## 📊 Workflow

### 1. Parameter-Tuning
Passe die Parameter in `dtw_engine.py` an und teste mit:
```bash
python test_robustness.py
```

### 2. Live-Test
Wenn Parameter gut funktionieren, teste mit echtem Audio:
```bash
python dtw_engine.py
```

### 3. Export für Teensy
Die optimierten Parameter werden dann in den Teensy-Code übertragen (C++ Implementierung).

## 🔧 Dependencies

```bash
pip install numpy librosa sounddevice scipy matplotlib
```

## 📝 Hinweise

- **ScoreData.h**: Wird vom MusescoreToChroma-Tool generiert
- **Live-Aufnahmen**: Können mit dem Live-Test aufgenommen und als .npy gespeichert werden
- **Tempo**: Alle Tests basieren auf BPM=40, BEATS_PER_MEASURE=4

## 🎼 Über das Projekt

Teil der Studienarbeit "Smarter Page Turner" - Ein intelligentes System zum automatischen Umblättern von Notenblättern für Musiker.

**Funktionsweise:**
1. Mikrofon nimmt Live-Musik auf
2. Chroma-Features werden extrahiert
3. ODTW vergleicht mit Referenz-Partitur
4. Position wird getrackt
5. Bei bestimmter Position → Bluetooth-Signal zum Umblättern

---

**Author:** Samuel Geffert
**Datum:** Februar 2026
