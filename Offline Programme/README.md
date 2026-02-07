# Offline Programme - Chroma Extraction Tools

Python-Tools zur Vorbereitung von Partitur-Daten für den Smarter Page Turner.

## 📁 Struktur

```
Offline Programme/
├── MusescoreToChroma/        # MusicXML → Chroma Konverter
├── PdfToChroma/              # PDF → Chroma (via OMR)
├── DTW_Studies/              # DTW-Algorithmus Tests & Entwicklung
│   ├── ODTW_Python/          # Online-DTW Implementierung ✨
│   ├── AudioToChroma/        # Live-Audio → Chroma
│   └── DTW-Simulationen.ipynb
├── data/                     # Zentrale Daten (NEU!)
│   ├── soundfonts/           # MuseScore_General.sf2 (206 MB)
│   ├── audio/                # Große .wav Dateien
│   ├── inputs/               # .musicxml Input-Dateien
│   └── generated/            # ScoreData.h, .mid Output
└── README.md                 # Diese Datei
```

## 🎯 Workflow

### 1. **Partitur → Chroma Konvertierung**

**Option A: Von MusicXML (empfohlen)**
```bash
cd MusescoreToChroma
python musescore_to_chroma.py ../data/inputs/Fiocco.musicxml --bpm 40 --instrument violin
# → Generiert ScoreData.h im aktuellen Ordner
```

**Option B: Von PDF (Optical Music Recognition)**
```bash
cd PdfToChroma
python pdf_to_chroma.py noten.pdf --bpm 40 --instrument violin
# → PDF wird via OMR in MusicXML konvertiert, dann Chroma berechnet
```

### 2. **Live-Audio → Chroma**
```bash
cd DTW_Studies/AudioToChroma
python AudioToChroma.py ../data/audio/Fiocco.wav --format npy
# → Erstellt Fiocco_chroma.npy
```

### 3. **DTW Testing**
```bash
cd DTW_Studies/ODTW_Python
python test_robustness.py  # Interaktive Tests
python dtw_engine.py        # Live-Test mit Mikrofon
```

## 📦 Dependencies

```bash
pip install numpy librosa sounddevice scipy matplotlib music21 oemer fluidsynth
brew install fluidsynth  # macOS
```

## 🗂️ data/ Ordner

Alle großen Dateien (Audio, Soundfonts, generierte Outputs) werden im `data/` Ordner gespeichert:

- **soundfonts/**: MuseScore_General.sf2 (206 MB) - Wird für MIDI-Synthese benötigt
- **audio/**: Große .wav Dateien (Fiocco.wav, Live-Aufnahmen)
- **inputs/**: .musicxml Partituren
- **generated/**: ScoreData.h und .mid Dateien (Output der Tools)

### Pfade anpassen

Die Skripte haben teilweise hardcodierte Pfade. Wenn du Probleme hast:

**MusescoreToChroma + PdfToChroma:**
- Zeile 10-12: `SOUNDFONT_PATH` auf `../data/soundfonts/MuseScore_General.sf2` ändern

**ODTW_Python:**
- Bereits aktualisiert, verwendet `data/` Unterordner ✓

## 🎼 Über die Tools

### MusescoreToChroma
Konvertiert MusicXML-Dateien zu Teensy-kompatiblen Chroma-Headern:
- Interaktive Seitenumbruch-Konfiguration
- MIDI-Synthese mit FluidSynth
- L2-normalisierte Chroma-Features
- Export als C-Header für Teensy

### PdfToChroma
Wie MusescoreToChroma, aber mit PDF-Input via OMR (Optical Music Recognition):
- Nutzt `oemer` für PDF → MusicXML
- Ansonsten identisch zu MusescoreToChroma

### DTW_Studies
Entwicklung und Testing des Online-DTW-Algorithmus:
- **ODTW_Python/**: Python-Prototyp mit umfangreichen Tests
- **AudioToChroma/**: Extrahiert Chroma aus Live-Aufnahmen
- **DTW-Simulationen.ipynb**: Jupyter Notebook für Visualisierungen

## ⚠️ Bekannte Issues

1. **Soundfont Pfad**: Manche Skripte haben absolute Pfade - bei Bedarf anpassen
2. **FluidSynth**: Muss installiert sein (`brew install fluidsynth`)
3. **Große Dateien**: .wav Dateien können schnell 50-100 MB werden

---

**Author:** Samuel Geffert
**Projekt:** Smarter Page Turner - Studienarbeit 2026
