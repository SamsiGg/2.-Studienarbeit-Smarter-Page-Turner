# Smarter Page Turner

Automatisches Notenblatt-Umblättern per Musikerkennung — entwickelt als Studienarbeit an der Hochschule Offenburg (2026).

Die vollständige wissenschaftliche Ausarbeitung ist hier im Repository enthalten:
**[Geffert, Samuel_4203964_Studienarbeit 6. Semester.pdf](Geffert,%20Samuel_4203964_Studienarbeit%206.%20Semester.pdf)**

---

## Idee

Ein Laptop hört per Mikrofon zu, während ein Musiker spielt. Ein Online-DTW-Algorithmus vergleicht das Gehörte in Echtzeit mit einer vorberechneten Referenz-Partitur und verfolgt so die aktuelle Position im Stück. Nähert sich der Musiker einem Seitenende, sendet das System ein MIDI-Signal per WLAN ans iPad — die Note-App blättert automatisch um.

---

## Struktur

```
Software/                   Kernsystem (Python)
├── Live Page Turner/       Echtzeit-Tracking: Mikrofon → ODTW → MIDI
├── Score Pipeline/         Partitur-Vorverarbeitung: MusicXML → .npz
├── data/                   Testdaten (Audio, Scores, generierte Grafiken)
└── ODTW_Python/            Evaluationsskripte & Visualisierungen für die Arbeit

Hardware/                   Eingebettete Systeme (C++ / PlatformIO)
├── Smarter Page Turner/    Teensy 4.1 Firmware (Audio + DTW)
└── Bluetooth-Manager/      ESP32-C3 Firmware (BLE Keyboard)
```

---

## Schnellstart

**Voraussetzungen:** Python 3.11+, `pip install librosa numpy sounddevice mido numba matplotlib`

**1. Partitur vorbereiten**
```bash
cd Software/Score\ Pipeline
python generate_score_data.py mein_stueck.musicxml --bpm 60
```

**2. Live-Tracking starten**
```bash
cd Software/Live\ Page\ Turner
python main.py --score ../data/scores/mein_stueck.npz
```

**3. iPad verbinden**
- Mac: Audio MIDI Setup → Netzwerk → Eigene Sitzung aktivieren
- iPad: Einstellungen → Musik → MIDI → Mac auswählen
- forScore: MIDI CC 64 = Blättern

---

## Technologie

| Schicht | Technologie |
|---|---|
| Chroma-Extraktion | librosa STFT (4096/512) |
| Tracking | Online-DTW mit Wait/Step/Skip, Suchfenster ±100 Frames |
| Recovery | Subsequence-DTW über Gesamtpartitur, Numba JIT |
| MIDI | Network MIDI (mido) |
| Parameter | Bayesianisch optimiert via Optuna |

---

Samuel Geffert · Hochschule Offenburg · 2026
