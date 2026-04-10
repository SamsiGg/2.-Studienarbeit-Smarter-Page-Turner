# Smarter Page Turner – Technische Dokumentation für die Studienarbeit

---

## 1. Überblick & Zielsetzung

Das System ermöglicht automatisches Seitenumblättern für Musiker während des Spielens.
Es verfolgt die aktuelle Position in einer Partitur in Echtzeit, indem es das Mikrofonsignal
kontinuierlich mit einer vorberechneten Referenz vergleicht – ohne menschliches Eingreifen.

**Zwei Hauptkomponenten:**
- **Score Pipeline** (Offline): Konvertiert eine Partitur (PDF oder MusicXML) in eine kompakte NPZ-Datei mit Referenz-Chroma-Vektoren und Seitenumbruch-Indizes.
- **Live Page Turner** (Echtzeit): Liest das Mikrofon-Audio, vergleicht es Frame für Frame mit der Referenz via Online-DTW und sendet ein MIDI-Signal zum Blättern.

---

## 2. Score Pipeline

### 2.1 OMR: PDF → MusicXML

Wenn die Partitur nur als PDF vorliegt, wird **Audiveris** als Optical Music Recognition (OMR) Engine eingesetzt. Audiveris erkennt Notenköpfe, Taktstriche, Vorzeichen etc. und erzeugt eine MusicXML-Datei.

**Bekannte Schwäche von OMR:**
- Dynamikangaben (pp, ff) werden häufig falsch oder gar nicht erkannt.
- Ungültige Notenschlüssel-Linien (z.B. F6, G6) kommen vor → werden automatisch auf gültige Standardlinien korrigiert.
- OMR-Ausgabe ist nie perfekt, aber für die Chroma-Extraktion ausreichend, da Dynamik ohnehin ignoriert wird (siehe Abschnitt 3.2).

**Designentscheidung:** Dynamik spielt für das Tracking keine Rolle, weil Kosinus-Distanz amplitudeninvariant ist (siehe Abschnitt 3.2). OMR-Fehler bei Dynamik sind daher irrelevant.

---

### 2.2 Audiosynthese: MusicXML → WAV

Die MusicXML wird zunächst via **music21** zu MIDI konvertiert und dann mit **FluidSynth** und einem Soundfont (MuseScore General SF2) zu einem WAV-File synthetisiert.

**Vorverarbeitung vor der Synthese:**
- **Grace Notes entfernen:** Grace Notes haben keine definierte Dauer → FluidSynth würde einen Dauerton erzeugen ("stuck note"). Sie werden vor dem MIDI-Export entfernt.
- **Tempo erzwingen:** Alle Tempo-Angaben in der MusicXML werden durch das gewünschte BPM ersetzt. Dies ist notwendig, weil der Musiker frei wählt, in welchem Tempo er übt.
- **Instrument setzen:** Das MIDI-Programm entspricht dem gespielten Instrument, um ein realistisches Klangbild zu erzeugen.

**Warum Synthese statt fixer Chroma-Tabelle?**

Ein synthetisierter Ton klingt über seine gesamte Dauer nicht gleich. Aufgrund der **ADSR-Hüllkurve** (Attack, Decay, Sustain, Release) ändert sich das Spektrum zeitlich. In der Attack-Phase überwiegen Obertöne, in der Sustain-Phase der Grundton. Eine feste Chroma-Zuordnung "Note X = Vektor Y" würde diesem zeitlichen Verlauf nicht gerecht. Die Synthese erzeugt automatisch das reale Chroma-Profil eines echt gespielten Tons.

---

### 2.3 Das Noise-Floor-Problem

**Problem:** FluidSynth produziert nach einem Ton einen reinen Reverb-Tail ohne Hintergraudrauschen. Dieser Nachhall klingt zwar leise aus, hat aber nach L2-Normierung dieselbe Chroma-Signatur wie der ursprüngliche Ton – weil keine anderen Geräusche ihn überdecken.

Im Live-Audio hingegen überlagert Raumrauschen den Nachhall sofort, sodass ein stiller Frame im Live-Signal wie Rauschen aussieht.

**Folge ohne Korrektur:** Der ODTW-Tracker sieht in der Referenz viele "Ton-Frames" nach jeder Note, aber im Live-Signal Rausch-Frames → schlechter Match → Tracker driftet.

**Lösung:** Vor der FFT wird dem synthetisierten Audio weißes Rauschen hinzugefügt:
```
audio_mit_rauschen = fluidsynth_audio + N(0, σ)
```
wobei σ = 0.004 so gewählt ist, dass es dem typischen Mikrofon-Rauschpegel entspricht.
Wenn der Ton leise genug ist (Reverb-Tail), dominiert das Rauschen → neutrales, inkonsistentes Chroma, das zum Live-Rauschen passt.

**Designentscheidung:** Diese Methode ist dem alternativen Ansatz (Schwellwert-Maske auf den leisesten X% der Frames) überlegen, weil:
- Sie kontinuierlich wirkt statt mit einem Hardcut.
- Sie stückunabhängig ist (kalibriert am absoluten Rauschpegel, nicht relativ).
- Sie die physikalische Realität direkt modelliert.

---

### 2.4 Chroma-Extraktion (Referenz)

**Parameter:**

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| Sample Rate | 44100 Hz | Standard-Audioqualität |
| FFT-Fenstergröße | 4096 Samples | ~93 ms Analysezeit |
| Hop-Size | 512 Samples | ~11,6 ms Update-Rate |
| Chroma-Bins | 12 | Halbtonklassen C bis H |

**Frequenzauflösung:**
```
Bin-Breite = 44100 / 4096 ≈ 10,77 Hz
```
Ausreichend für Halbtonklassen (Abstand zweier Halbtöne bei A4: 26 Hz).

**Zeitauflösung trotz 93ms-Fenster:**
Durch Hop-Size 512 wird alle 11,6 ms ein neuer Frame berechnet (87,5% Überlappung). Bei 40 BPM mit Sechzehntelnoten (375 ms) ergibt das ~32 Frames pro Note – robust genug für zuverlässiges Tracking.

**Chroma-Extraktion im Detail (STFT-basiert):**
1. FFT über 4096-Sample-Fenster → Frequenzspektrum
2. Jeder Frequenzbin wird der nächsten Halbtonklasse zugeordnet: `midi = 12 * log2(f/440) + 69`, dann `chroma = midi % 12`
3. Energie aller Bins derselben Klasse wird summiert (inkl. Obertöne)
4. L2-Normalisierung: `chroma = chroma / ||chroma||₂`

**Warum L2-Normierung?**
Die L2-Norm entspricht der geometrischen Länge eines Vektors (Pythagoras). Nach Division durch diese Länge hat der Vektor die Länge 1 und liegt auf der Einheitskugel. Das ist Voraussetzung für effiziente Kosinus-Distanzberechnung (siehe Abschnitt 3.2).

---

### 2.5 Seitenumbrüche

Die Seitenumbrüche werden aus der MusicXML ausgelesen (`<print new-page="yes"/>`). Für jeden Seitenumbruch wird der Frame-Index des letzten Takts der jeweiligen Seite berechnet:

```
frames_per_beat = (60 / bpm) * (sample_rate / hop_length)
frame_index = beat_offset_in_quarter_notes * frames_per_beat
```

**Wichtig:** music21 arbeitet intern immer in Viertelnoten-Offsets, unabhängig von der Taktart. Ein Taktwechsel (z.B. 4/4 → 3/4) beeinflusst die Frame-Berechnung deshalb nicht – der Offset in Viertelnoten bleibt korrekt.

---

### 2.6 Das NPZ-Archiv

Die fertigen Daten werden als NumPy-Archiv (.npz) gespeichert:

| Feld | Typ | Inhalt |
|------|-----|--------|
| `chroma` | float32 (12 × N) | L2-normalisierte Referenz-Chroma-Matrix |
| `page_end_indices` | int32 (P,) | Frame-Index des Seitenendes für jede Seite |
| `bpm` | int32 | Tempo |
| `beats_per_measure` | int32 | Taktart-Zähler |
| `musicxml_content` | str | Originalpartitur für optionale Notendarstellung |

---

## 3. Live Page Turner

### 3.1 Architektur & Threading

Das System verwendet zwei Threads, die über eine Queue kommunizieren:

**Worker-Thread** (Audio & Tracking):
- Liest alle 11,6 ms einen Block (512 Samples) vom Mikrofon
- Schreibt in einen 4096-Sample-Ringbuffer
- Berechnet Chroma aus dem vollen Buffer (Sliding Window)
- Führt einen ODTW-Schritt aus
- Schreibt den TrackerState in eine Queue

**Main-Thread** (GUI):
- Pollt die Queue alle 50 ms (20 Hz)
- Nimmt nur den neuesten State (ältere verwerfen)
- Aktualisiert GUI-Elemente

**Warum Queue statt direktem Zugriff?**
tkinter ist nicht thread-safe. Direkte GUI-Manipulation aus dem Worker-Thread würde zu Crashes führen. Die Queue ist intern thread-safe durch ein Lock.

**Ringbuffer:**
```
Neuer Block (512) → in Buffer schieben → älteste 512 Samples raus
Voller Buffer (4096) → FFT → Chroma
```
Effekt: Jeder Chroma-Frame repräsentiert die letzten ~93 ms Audio, aktualisiert alle ~11,6 ms.

---

### 3.2 Distanzmaß: Kosinus-Distanz

**Warum nicht euklidische Distanz?**

Ein leise gespielter C-Dur-Akkord hat dieselbe Chroma-Richtung wie ein laut gespielter C-Dur-Akkord, aber eine andere Länge (Amplitude). Euklidische Distanz würde einen mittellauten Fis-Dur-Akkord als näher am leisen C-Dur einordnen als ein laut gespielter C-Dur.

Kosinus-Distanz misst nur den **Winkel** zwischen Vektoren – sie ist amplitudeninvariant:
```
d_kosinus = 1 - (a · b) / (||a|| · ||b||)
```

Nach L2-Normierung vereinfacht sich dies zu:
```
d_kosinus = 1 - (a · b)    [da ||a|| = ||b|| = 1]
```
Nur noch ein Skalarprodukt – kein Wurzelziehen, effizient für Echtzeit.

**Mathematischer Hintergrund:** Das Skalarprodukt ist als "Dot Product" aus dem Schulunterricht bekannt. Die Kosinus-Ähnlichkeit ist das Skalarprodukt zweier normierter Vektoren.

**Bekannte Schwäche:** Bei perfekt reinen Einzeltönen (nur ein Chroma-Bin ≠ 0) haben alle anderen Töne denselben Abstand (90°). In der Praxis spielt das kaum eine Rolle, da echte Töne immer Obertöne haben und mehrere Bins anregen.

---

### 3.3 Online-DTW (ODTW)

#### Grundprinzip DTW

Dynamic Time Warping (DTW) findet die optimale Ausrichtung zwischen zwei Zeitreihen, auch bei Tempoabweichungen. Im Gegensatz zu euklidischer Distanz über Zeit erlaubt DTW, dass ein Frame der Live-Sequenz mehreren Frames der Referenz zugeordnet wird (bei langsamem Spielen) oder übersprungen wird (bei schnellem Spielen).

**Warum keine fertige DTW-Bibliothek?**
Alle verfügbaren Python-Bibliotheken (librosa, dtw-python, tslearn) implementieren nur **offline** DTW – beide Sequenzen müssen vollständig vorliegen. Für Echtzeit-Tracking ist das nicht möglich. Eine Online-DTW-Implementierung existiert als fertige Bibliothek nicht, weshalb sie selbst implementiert wurde.

#### Online-Betrieb

Bei Online-DTW liegt die Referenz vollständig vor, aber die Live-Sequenz wächst Frame für Frame. Für jeden neuen Live-Frame werden die akkumulierten Kosten für alle Referenz-Positionen in einem Suchfenster aktualisiert.

**Drei Pfad-Optionen pro Frame:**

| Pfad | Bedeutung | Penalty |
|------|-----------|---------|
| Wait (vertikal) | Aktueller Live-Frame → gleiche Ref-Position | 0.1 |
| Step (diagonal) | Aktueller Live-Frame → nächste Ref-Position | 0.0 |
| Skip (diagonal ×2) | Aktueller Live-Frame → übernächste Ref-Position | 0.03 |

**Damping-Faktor (0.99):**
Akkumulierte Kosten werden mit 0.99 multipliziert bevor ein neuer Frame addiert wird. Ohne Damping würden frühere Pfadfehler ewig akkumuliert und der Tracker könnte nicht mehr von schlechten Stellen erholen. Mit Damping verlieren alte Kosten graduell ihren Einfluss (0.99^300 ≈ 0.05).

**Suchfenster (±100 Frames, ~1,2 Sekunden):**
Statt den gesamten Referenz zu scannen, wird nur ein Fenster um die aktuelle Position betrachtet. Das begrenzt Rechenlast und verhindert, dass der Tracker in weit entfernte Partiturbereiche springt.

---

### 3.4 Recovery-Mechanismus

#### Problem

Wenn ein Musiker eine Wiederholung spielt, eine Passage wiederholt oder stark vom Tempo abweicht, kann der ODTW-Tracker die Position verlieren. Er "driftet" in einen Bereich der Referenz, der nicht dem aktuell Gespielten entspricht.

#### Erkennung

Ein gleitender Mittelwert der Tracking-Kosten über 300 Frames (~3,5 Sekunden) wird berechnet. Übersteigt dieser Wert den Schwellwert (38), ist ein Recovery nötig.

Warum gleitender Mittelwert? Ein einzelner schlechter Frame (Außengeräusch, Spielfehler) soll keinen unnötigen Recovery auslösen. Erst wenn der Tracker über mehrere Sekunden schlecht matched, wird eingegriffen.

#### Subsequence-DTW-Scan

Beim Recovery werden die letzten ~3,5 Sekunden Live-Audio (300 Frames) gegen die gesamte Referenz gescannt, um die wahrscheinlichste aktuelle Position zu finden.

**Wichtig: Kein einfacher linearer Scan.**
Ein linearer Frame-für-Frame-Vergleich versagt bei Tempoabweichungen. Wenn der Musiker 30% schneller spielt als die Referenz, sind dieselben Töne auf weniger Frames verteilt → ein festes Vergleichsfenster passt nicht.

**Lösung: Subsequence-DTW**
Ein kleines DTW zwischen den 300 Live-Frames und jedem Abschnitt der Referenz findet die beste Ausrichtung unabhängig vom Tempo. Der Scan subsampled die Referenz (jeden 10. Frame) für Effizienz und nutzt Numba JIT-Kompilierung für die inneren Schleifen.

**Numba:**
Die verschachtelten Schleifen des DTW-Kerns werden beim ersten Aufruf von Numba in nativen Maschinencode kompiliert (LLVM). Danach 50–100× schneller als reines Python – der Unterschied zwischen ~2 Sekunden und ~20 ms für den Scan.

---

### 3.5 Seitenumblätter-Logik

**PAGE_TURN_OFFSET (100 Frames, ~1,2 Sekunden):**
Der Blätterbefehl wird gesendet, wenn der Tracker 100 Frames vor dem gespeicherten Seitenende ankommt. Das wird getan, da Musiker die Noten immer ein wenig vorauslesen.

**MIDI-Signal (nicht so wichtig):**
Ein MIDI Control Change Message (CC 64, Sustain Pedal) wird über Bluetooth MIDI gesendet. Dieser Standard-CC-Wert wird von Noten-Apps (forScore, Newzik) nativ als "Nächste Seite" interpretiert.

---

### 3.6 Sleep-Modus & Start-Erkennung

**Start:** Der Tracker beginnt erst zu laufen, wenn das Mikrofon-Signal einen RMS-Schwellwert (0.01) überschreitet. Hintergraudrauschen löst kein Tracking aus.

**Sleep:** Nach 500 aufeinanderfolgenden stillen Frames (~5,8 Sekunden) wechselt der Tracker in den Sleep-Modus und pausiert die Berechnung. Bei neuem Audio wacht er wieder auf.

---

## 4. Designentscheidungen & Abwägungen

### Offline-Vorbereitung vs. Echtzeit

Die Vorab-Berechnung der Referenz-Chroma offline ist eine zentrale Designentscheidung. Alternativen wären:
- **Echtzeit-Parsing der MusicXML:** Zu aufwändig, erfordert Audiosynthese in Echtzeit.
- **Statische Chroma-Tabellen:** Ignoriert ADSR und Obertöne → schlechterer Match.
- **Echte Referenzaufnahme:** Erfordert eine professionelle Referenzeinspielung in exakt dem gewünschten Tempo → unpraktisch.

Die gewählte Syntheselösung erlaubt beliebige Tempi und Instrumente mit einmaliger Offline-Vorbereitung.

### Chroma vs. andere Features

Chroma-Vektoren sind für diesen Anwendungsfall ideal:
- **Oktavinvarianz:** A2 und A4 haben denselbe Chroma-Vektor → vereinfacht Matching.
- **Robustheit gegen Timbre:** Ein Geigen-A und ein Klavier-A haben ähnliche Chroma, aber sehr verschiedene Spektren.
- **Kompakt:** 12 Werte pro Frame, geringe Rechenlast.

Alternativen wie MFCCs (Mel Frequency Cepstral Coefficients) würden Timbre stärker gewichten und wären empfindlicher gegenüber Instrument-Unterschieden zwischen Referenz und Live-Audio.

### Numba vs. C-Extension vs. reines NumPy (keine wichtige Entscheidung)

Für den Recovery-Scan:
- **Reines NumPy:** Nicht möglich, da DTW sequenzielle Abhängigkeiten hat (jeder Schritt hängt vom vorherigen ab) – nicht vektorisierbar.
- **C-Extension:** Maximal schnell, aber plattformabhängig und komplex zu kompilieren.
- **Numba:** Schnell wie C (nach JIT-Warmup), reines Python bleibt lesbar, automatisch kompatibel mit NumPy-Arrays. Optimaler Kompromiss für dieses Projekt.

---

## 5. Parameter-Übersicht (empirisch optimiert)

| Parameter | Wert | Begründung |
|-----------|------|-----------|
| FFT-Fenstergröße | 4096 Samples | Balance zwischen Frequenz- und Zeitauflösung |
| Hop-Size | 512 Samples | ~86 Hz Update-Rate, ausreichend für 40 BPM |
| Suchfenster | ±100 Frames | ~1,2 s Spielraum für Tempovariationen |
| Damping | 0.99 | Gedächtnis ca. 100 Frames (~1,2 s) |
| Wait-Penalty | 0.1 | Bevorzugt Vorwärtsbewegung |
| Skip-Penalty | 0.03 | Erlaubt leichtes Beschleunigen |
| Recovery-Schwelle | 38 | Über 3,5 s gemittelte Kosten |
| Noise-Floor-σ | 0.004 | Entspricht typischem Mikrofon-Rauschpegel |
| Page-Turn-Offset | 100 Frames | ~1,2 s Vorsprung zum Blättern |

**Parameterwahl-Methodik:**
Die Parameter wurden empirisch durch zwei Validierungsmethoden bestimmt: Erstens durch Simulation des ODTW-Trackers mit echten Live-Aufnahmen als Eingabe, wodurch Tracking-Verhalten und Recovery-Auslösung unter realen Bedingungen beobachtet werden konnten. Zweitens durch visuelle Chroma-Vergleiche zwischen synthetisierter Referenz und Live-Audio über mehrere Takte, um Synthesefehler und Abweichungen im Referenz-Chroma frühzeitig zu erkennen.

---

## 6. Mögliche Verbesserungen (Ausblick)

- **Mikrocontroller-Portierung:** Der ODTW-Kern besteht im Wesentlichen aus FFT, 12 Multiplikationen, 12 Additionen und einer DP-Rekurrenz – alles auf einem ARM-Mikrocontroller mit FPU realisierbar. Chroma-Extraktion benötigt nur die FFT und eine Lookup-Tabelle (Bin → Chroma-Klasse).

- **Präziseres Noise-Modeling:** Anstatt eines fixen σ könnte das Rauschen einmalig vom Mikrofon des Nutzers gemessen werden (Kalibrierungsschritt bei Stille), um den Rauschpegel exakt anzupassen.

- **Taktwechsel-Unterstützung in der GUI:** Taktart und Schlaganzeige sind aktuell fix. Eine Liste von Taktwechseln (aus music21 auslesbar) würde die Taktanzeige bei Stücken mit wechselnden Taktarten korrekt machen.

- **Adaptives Tempo:** Aktuell ist BPM fest vorgegeben. Eine Tempo-Schätzung aus dem Live-Audio (Onset-Detektion) könnte den Tracker robuster gegenüber globalem Tempodrift machen.

- **Mehrere Soundfonts / Instrumente:** Für Kammermusik könnten mehrere Instrumente gleichzeitig synthetisiert werden (z.B. Klavier-Begleitung + Solovioline), um das Chroma realistischer zu gestalten.

- **Verbesserte Recovery-Effizienz:** Der Subsequence-DTW-Scan könnte hierarchisch arbeiten (erst grobe Suche mit starkem Subsampling, dann Verfeinerung) für noch schnellere Recovery bei langen Partituren.

---

## 7. Empfehlung für die Gliederung der Studienarbeit

### Tiefgründig behandeln (eigener Beitrag, Kernthemen):
- Online-DTW: Algorithmus, Pfad-Optionen, Damping-Faktor, Suchfenster
- Recovery-Mechanismus: Erkennung, Subsequence-DTW, Sprungbedingung
- Noise-Floor-Problem und Lösung
- Chroma-Extraktion: STFT, Bin-zu-Chroma-Mapping, L2-Normierung
- Kosinus-Distanz vs. euklidische Distanz: Amplitudeninvarianz, Effizienz

### Mittel behandeln (notwendiger Kontext):
- Score Pipeline: OMR, Audiosynthese, ADSR-Motivation
- Seitenumbruch-Berechnung via Viertelnoten-Offset
- Threading-Modell und Ringbuffer
- MIDI-Signalgebung

### Nur oberflächlich (bekannte Grundlagen, mit Quellen):
- DTW (allgemein) – viele Veröffentlichungen vorhanden
- FFT und STFT – Standardlehrstoff
- Chroma-Features in der Musikinformatik – etabliert seit Ellis & Poliner (2007)
- Score Following als Forschungsfeld – Grundlagen aus Dixon (2005), Raphael (2010)
- MIDI-Protokoll
- Soundfont-Synthese / FluidSynth