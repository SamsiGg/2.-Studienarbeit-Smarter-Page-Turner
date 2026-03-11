# Smarter Page Turner – Projektübersicht

---

## 1. Score Pipeline

### Zweck
Wandelt eine Partitur (PDF oder MusicXML) in eine `.npz`-Datei um, die alle für das Live-Tracking nötigen Daten enthält. Dieses vorberechnete Format macht das Live-System schlank und echtzeitfähig.

### Ablauf (3 Stufen)

```
PDF  →  [Audiveris OMR]  →  MusicXML
MusicXML  →  [music21 + FluidSynth]  →  WAV-Audio
WAV  →  [librosa STFT]  →  Chroma-Matrix  →  .npz
```

---

### Stufe 1 – OMR (nur bei PDF-Input)

- **Tool:** Audiveris (Open Source, lokal)
- Wird automatisch in `/Applications/Audiveris.app` oder via `AUDIVERIS_PATH`-Umgebungsvariable gefunden
- Ausgabe: `.mxl`-Datei (gezipptes MusicXML)

---

### Stufe 2 – Audio-Synthese & Chroma-Extraktion (`chroma_builder.py`)

Dies ist das Herzstück der Pipeline und enthält mehrere maßgeschneiderte Features:

#### 2a – Partitur-Vorverarbeitung
- Laden der MusicXML via **music21**
- **Grace Notes entfernen:** Schlag-Noten ohne echte Dauer verursachen „Stuck Notes" in FluidSynth (MIDI-Noten ohne Note-Off). Sie sind für die Chroma-Erkennung irrelevant und werden deshalb entfernt.
- **Tempo erzwingen:** Alle bestehenden Tempo-Markierungen werden gelöscht und durch den gewünschten BPM-Wert ersetzt. Das Stück wird so synthetisiert wie es live gespielt wird.
- **Instrument setzen:** Über MIDI-Programm-Nummern wählbar (`violin=40`, `piano=0`, `cello=42`, ...).

#### 2b – Audio-Synthese via FluidSynth
- MIDI-Export aus music21 → temporäre `.mid`-Datei
- FluidSynth synthetisiert das Audio mit einem Soundfont (`MuseScore_General.sf2`):
  ```
  fluidsynth -ni -g 1.0 -F output.wav -r 44100 soundfont.sf2 score.mid
  ```
- Ausgabe: 44100 Hz Mono-WAV

**Warum Synthese statt realer Aufnahme?**
Das Live-System erkennt die Noten per Chroma. Damit Referenz und Live-Audio vergleichbar sind, muss die Referenz unter denselben akustischen Bedingungen (Instrument, Tempo) erzeugt werden wie das echte Spiel.

#### 2c – Chroma-Extraktion
- **STFT** via librosa: FFT-Fenstergröße 4096, Hop-Length 512 (≈ 11,6 ms pro Frame)
- 12 Chroma-Bins (C, C#, D, ..., B) – eine Dimension pro Halbton
- Ergebnis: Matrix (12 × N) – für jeden Zeitframe ein 12-dimensionaler Vektor

#### 2d – Stille-Erkennung & Masking (Custom Feature)
- RMS-Energie pro Frame berechnen
- Frames unterhalb des **3. Perzentils** gelten als still (Pausen, Reverb-Ausklang)
- Stille Frames → **gleichmäßig verteiltes Zufallsrauschen** statt Null-Vektor
- **Warum?** Ein Null-Vektor bei L2-Normalisierung ist numerisch instabil. Zufallsrauschen im Pausenbereich verhindert, dass der Tracker bei Pausen in eine falsche Position zieht.
- `silence_mask`: Bool-Array, das markiert welche Frames still sind → wird im Live-System genutzt, um Recovery-Sprünge in stille Bereiche zu blockieren.

#### 2e – L2-Normalisierung
- Jeder Chroma-Vektor wird auf Länge 1 normiert
- Macht die Kosinusähnlichkeit unabhängig von der Lautstärke
- Entspricht dem Vorgehen im Live-System

---

### Stufe 3 – Seitenumbrüche & Speicherung

#### Seitenumbruch-Erkennung (Custom Feature)
- **Automatisch:** MusicXML enthält `<print new-page="yes"/>` Attribute. Diese werden geparst, die entsprechenden Takt-Nummern in Frame-Indizes umgerechnet:
  ```
  frames_per_beat = (60 / BPM) * (sample_rate / hop_length)
  frame_index = beat_offset * frames_per_beat
  ```
- **Interaktiver Fallback:** Falls keine Seitenumbrüche gefunden werden, fragt das Script den Nutzer nach der letzten Taktnummer jeder Seite.
- **Bestätigungsdialog:** Erkannte Seitenumbrüche werden immer angezeigt und müssen mit `y` bestätigt oder mit `n` manuell überschrieben werden.

#### NPZ-Dateiformat
| Feld | Typ | Inhalt |
|------|-----|--------|
| `chroma` | `float32 (12, N)` | L2-normierte Chroma-Matrix |
| `page_end_indices` | `int32 (K,)` | Frame-Indizes an Seitengrenzen |
| `silence_mask` | `bool (N,)` | True = stiller Frame |
| `bpm` | `int32` | Tempo |
| `beats_per_measure` | `int32` | Taktart (z.B. 4 für 4/4) |
| `musicxml_content` | `str` | Vollständige MusicXML-Quelle |

---

## 2. Live Page Turner

### Zweck
Liest Mikrofon-Audio in Echtzeit, extrahiert Chroma-Features und verfolgt die Position in der Partitur per Online-DTW. Bei erkanntem Seitenende wird ein MIDI-Signal gesendet.

### Systemarchitektur

```
Mikrofon  →  [AudioRingBuffer]  →  [ChromaExtractor]  →  [ODTWTracker]
                                                               ↓
                                                    Position / Seite / Beat
                                                               ↓
                                                     [MIDI via Network MIDI]
                                                               ↓
                                                         iPad (forScore)
```

- **Main-Thread:** tkinter GUI
- **Worker-Thread:** Mikrofon → RingBuffer → Chroma → ODTW → Queue
- Kommunikation über `queue.Queue` (maxsize=5, älteste Frames werden verworfen)

---

### Chroma-Extraktion Live (`chroma.py`)

- **AudioRingBuffer:** Kreispuffer mit 4096 Samples. Neue Samples werden hinten angehängt, alte vorne verworfen.
- **ChromaExtractor:** Identische STFT-Parameter wie in der Pipeline (FFT=4096, Hop=4097-Trick für genau 1 Output-Frame, kein Auto-Tuning).
- **Warum kein Auto-Tuning?** Das Teensy-System hat ebenfalls kein Auto-Tuning. Konsistenz zwischen Embedded- und Desktop-System ist wichtiger als absolute Frequenzgenauigkeit.

---

### ODTW-Algorithmus (`dtw.py`)

#### Grundprinzip
Online Dynamic Time Warping vergleicht jeden eingehenden Live-Chroma-Frame mit einer Umgebung um die aktuelle Position in der Referenz. Drei Übergänge sind erlaubt:

| Übergang | Bedeutung | Strafe |
|----------|-----------|--------|
| Wait | Tracker bleibt stehen (Spieler wiederholt) | `penalty_wait = 0.1` |
| Step | Tracker geht 1 Frame vor (normal) | `penalty_step = 0.0` |
| Skip | Tracker springt 2 Frames (Spieler beschleunigt) | `penalty_skip = 0.03` |

Kostenformel pro Frame `i` im Suchfenster:
```
cost[i] = distance(live, ref[i]) + damping * min(
    cost[i]   + penalty_wait,   # warten
    cost[i-1] + penalty_step,   # 1 vor
    cost[i-2] + penalty_skip    # 2 vor
)
```

- `distance` = Kosinus-Distanz: `1 - cos_similarity(live, ref[i])`
- `damping = 0.99`: Alte Kosten werden pro Frame mit 0.99 multipliziert (Bayesianisch optimiert). Verhindert, dass sehr alte Fehler die aktuelle Entscheidung dominieren.
- **Suchfenster:** Nur ±100 Frames um die aktuelle Position werden geprüft – O(search_window) statt O(N).

#### Start-Erkennung
- Tracking startet erst wenn RMS > `START_THRESHOLD_RMS = 0.01`
- Verhindert falsches Tracking bei Stille vor dem Stück

#### Sleep-Modus
- Nach 500 aufeinanderfolgenden stillen Frames → Tracker schläft
- Weckt sich selbst wenn RMS wieder über den Schwellwert steigt
- Position wird bei Stille nicht weiter verfolgt → kein Drift

---

### Recovery-Mechanismus (Custom Feature)

Der Recovery-Mechanismus erkennt, wenn der Tracker die Spur verloren hat (z.B. nach einer Unterbrechung, einem Fehler oder einem langen Sprung im Stück), und repositioniert ihn automatisch.

#### Trigger
- Rollierender Durchschnitt der letzten 300 Kosten-Werte wird berechnet
- Wenn `avg_cost > recovery_threshold (=38)` → Recovery wird gestartet

#### Ablauf (Full-Score Scan)
1. **Live-History sammeln:** Letzte 300 Chroma-Frames aus dem internen Puffer
2. **Distanzmatrix berechnen:** Kosinus-Distanz zwischen jedem Live-Frame und jedem (10. subgesampelten) Referenz-Frame → Matrix (300 × N/10)
3. **Stille maskieren:** Spalten, die stillen Referenz-Frames entsprechen → Distanz = ∞ (verhindert Sprung in Pausen)
4. **Subsequence-DTW:** Dynamische Programmierung über die Distanzmatrix. Freier Start (`dp[0,:] = 0`), um beliebige Position im Score zu finden.
5. **Qualitätsbewertung:** Mittlere Kosinus-Distanz am besten gefundenen Punkt
   - 0.0 = perfekte Übereinstimmung
   - 1.0 = keine Übereinstimmung / Stille
6. **Sprungentscheidung:** Wenn `quality < recovery_jump_threshold (=1.0)` → Tracker springt an neue Position

#### Numba JIT-Beschleunigung
- Der Subsequence-DTW-Kern ist mit `@njit` (Numba) dekoriert
- Erste Ausführung: ~1-3s Kompilierung; danach nahezu C-Geschwindigkeit
- Fallback auf NumPy wenn Numba nicht verfügbar

---

### Seitenumblättern

- `page_end_indices` aus der NPZ-Datei definieren Seitengrenzen
- Trigger bereits **10 Frames vor** dem eigentlichen Seitenende (`PAGE_TURN_OFFSET = 10`)
- Nach einem Recovery-Sprung wird eine Karenzzeit abgewartet (`PAGE_TURN_STABLE_TIME`), bevor ein Seitenumblättern gesendet wird

---

### MIDI-Ausgabe (`ble_keyboard.py`)

- Nutzt **Network MIDI** (WLAN) über macOS Audio MIDI Setup
- Sendet MIDI Control Change Nachrichten an iPad:
  - CC 64 (Sustain Pedal) → Seite vor (Standard für forScore)
  - CC 67 (Soft Pedal) → Seite zurück
- Automatische Port-Erkennung ("Network" / "Netzwerk")

**Einrichtung:**
1. Mac: Audio MIDI Setup → Netzwerk → Eigene Sitzung aktivieren
2. iPad: Einstellungen → Musik → MIDI → Mac auswählen
3. forScore: MIDI-Einstellungen → CC 64 = Blättern

---

## 3. Parameter-Optimierung (Hintergrund)

Die ODTW-Parameter (`damping_factor`, `wait_penalty`, `skip_penalty`, `recovery_threshold`, etc.) wurden **Bayesianisch optimiert** via Optuna. Dafür wurden Testaufnahmen mit bekannter Ground-Truth-Position aufgenommen und die Parameter so gewählt, dass der mittlere Tracking-Fehler minimiert wird. Die optimierten Werte sind in `settings.py` fest eingetragen.

---

## 4. Technologie-Stack

| Komponente | Technologie |
|------------|------------|
| Noten-Parsing | music21 |
| OMR (PDF→XML) | Audiveris |
| Audio-Synthese | FluidSynth + MuseScore Soundfont |
| Chroma-Extraktion | librosa (STFT) |
| DTW-Kern | NumPy / Numba JIT |
| Audio-Input | sounddevice |
| MIDI-Ausgabe | mido + python-rtmidi |
| GUI | tkinter |
| Dateiformat | NumPy NPZ |
