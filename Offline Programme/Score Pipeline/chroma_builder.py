# =============================================================================
# chroma_builder.py – Chroma-Vektoren direkt aus MusicXML berechnen
# =============================================================================
# Kern-Innovation: Kein Audio nötig!
#
# Alte Pipeline:  MusicXML → MIDI → FluidSynth (Audio) → librosa (Chroma)
# Neue Pipeline:  MusicXML → music21 (Noten parsen) → direkt Chroma berechnen
#
# Chroma-Vektoren sind Tonklassen-Verteilungen (C, C#, D, ..., B).
# Wenn wir die Noten kennen, können wir die Verteilung direkt berechnen,
# ohne den Umweg über Audio-Synthese + FFT.
# =============================================================================

import numpy as np
import music21

# Audio-Parameter (müssen mit Live Page Turner / Teensy übereinstimmen)
SAMPLE_RATE = 44100
HOP_LENGTH = 512

# Geigen-Chroma-Template: empirisch gemessen aus einer echten Live-Aufnahme
# (Fiocco Allegro, Violine, 40 BPM).
#
# Methode: Für jeden Chroma-Frame den dominanten Ton finden, die Chroma-Verteilung
# zum Grundton zentrieren, über ~44000 Frames mitteln.
#
# Enthält Obertöne (Quinte, Terz) UND Vibrato-Spread (±1 Halbton).
#                    C      C#     D      D#     E      F      F#     G      G#     A      A#     B
#                   Grund  +1HT  +2Sek  +3mT   +4gT   +5Qu   +6Tri  +7Qui  +8kS   +9gS   +10kS  +11gS
VIOLIN_TEMPLATE = [1.0000, 0.2133, 0.0314, 0.0371, 0.0818, 0.0493, 0.0453, 0.1767, 0.0539, 0.0320, 0.0383, 0.2110]

# Einschwingzeit in Frames (~50ms bei 86 Hz Frame-Rate)
ATTACK_FRAMES = 5
# Ausschwingzeit in Frames (~70ms)
DECAY_FRAMES = 6
# Rausch-Level: kleine Variation damit ODTW Frames unterscheiden kann
NOISE_LEVEL = 0.02
# Hintergrund-Level für Pausen (simuliert Raumrauschen)
REST_NOISE_LEVEL = 0.005


def build_chroma(musicxml_path: str, bpm: int,
                 sample_rate: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH) -> tuple[np.ndarray, list[int]]:
    """MusicXML → Chroma-Matrix + Seitenumbruch-Indizes.

    Berechnet 12-dimensionale Chroma-Vektoren direkt aus den Notendaten,
    ohne Audio-Synthese. Simuliert realistische Klang-Charakteristiken:
    - Obertöne (empirisch gemessenes Geigen-Template)
    - Einschwing-/Ausschwing-Rampen (Attack/Decay)
    - Leichte Frame-zu-Frame Variation (Noise)
    - Hintergrundpegel bei Pausen

    Args:
        musicxml_path: Pfad zur .musicxml oder .mxl Datei.
        bpm: Tempo in BPM (überschreibt Tempo-Markierungen in der Datei).
        sample_rate: Samplerate (muss mit Live-System übereinstimmen).
        hop_length: Hop-Length (muss mit Live-System übereinstimmen).

    Returns:
        (chroma, page_end_indices):
            chroma: Shape (12, N), L2-normalisiert.
            page_end_indices: Liste der Frame-Indizes an Seitenenden.
    """
    print(f"Lade {musicxml_path}...")
    score = music21.converter.parse(musicxml_path)

    if not score.parts:
        raise ValueError("Keine Stimmen in der Partitur gefunden!")

    part = score.parts[0]

    # Tempo erzwingen (alle vorhandenen Tempomarkierungen entfernen)
    _set_tempo(part, bpm)

    # Zeitumrechnung: quarter-note offsets → Frame-Indizes
    sec_per_beat = 60.0 / bpm
    frames_per_beat = sec_per_beat * (sample_rate / hop_length)

    # Gesamtlänge
    total_beats = part.duration.quarterLength
    total_frames = int(total_beats * frames_per_beat) + 1
    print(f"  Tempo: {bpm} BPM, Dauer: {total_beats:.1f} Beats, {total_frames} Frames")

    # Geigen-Template (empirisch aus Live-Aufnahme gemessen)
    base_template = np.array(VIOLIN_TEMPLATE, dtype=np.float32)
    print(f"  Geigen-Template (empirisch): Grundton=1.00, Quinte={base_template[7]:.2f}, Vibrato={base_template[1]:.2f}/{base_template[11]:.2f}")

    # Chroma-Matrix: (12, total_frames)
    chroma = np.zeros((12, total_frames), dtype=np.float32)

    note_count = 0
    for element in part.recurse().notes:
        # music21 .notes gibt Notes UND Chords zurück
        if element.isChord:
            pitches = [p.midi for p in element.pitches]
        else:
            pitches = [element.pitch.midi]

        onset_beats = element.getOffsetInHierarchy(part)
        duration_beats = element.duration.quarterLength

        start_frame = int(onset_beats * frames_per_beat)
        end_frame = int((onset_beats + duration_beats) * frames_per_beat)

        # Clamp auf gültige Indizes
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        n_frames = end_frame - start_frame

        # Amplituden-Hüllkurve: Attack → Sustain → Decay
        envelope = np.ones(n_frames, dtype=np.float32)
        # Attack-Rampe (linear von 0.1 auf 1.0)
        attack_len = min(ATTACK_FRAMES, n_frames // 2)
        if attack_len > 0:
            envelope[:attack_len] = np.linspace(0.1, 1.0, attack_len)
        # Decay-Rampe (linear von 1.0 auf 0.3)
        decay_len = min(DECAY_FRAMES, n_frames // 2)
        if decay_len > 0:
            envelope[-decay_len:] = np.linspace(1.0, 0.3, decay_len)

        for midi_pitch in pitches:
            # Template zum Grundton rotieren
            chroma_bin = midi_pitch % 12
            rotated = np.roll(base_template, chroma_bin)
            # Mit Hüllkurve multiplizieren (jeder Frame hat leicht andere Amplitude)
            chroma[:, start_frame:end_frame] += rotated[:, np.newaxis] * envelope[np.newaxis, :]
            note_count += 1

    print(f"  {note_count} Noten-Events verarbeitet")

    # Leichte Zufallsvariation hinzufügen (verhindert identische Frames)
    rng = np.random.default_rng(42)  # Reproduzierbar
    noise = rng.normal(0, NOISE_LEVEL, chroma.shape).astype(np.float32)
    chroma += np.abs(noise)  # Nur positive Werte (Chroma kann nicht negativ sein)

    # Hintergrund-Rauschen für Pausen (Zero-Frames vermeiden)
    frame_energy = np.linalg.norm(chroma, axis=0)
    silent_mask = frame_energy < 0.01
    n_silent = np.sum(silent_mask)
    if n_silent > 0:
        bg_noise = np.abs(rng.normal(0, REST_NOISE_LEVEL, (12, n_silent))).astype(np.float32)
        chroma[:, silent_mask] = bg_noise
        print(f"  {n_silent} Pausen-Frames mit Hintergrundrauschen gefüllt")

    # L2-Normalisierung pro Frame
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    chroma = chroma / (norms + 1e-9)

    # Interaktive Seitenumbrüche
    page_indices = _get_interactive_page_turns(score, frames_per_beat)

    return chroma, page_indices


def _set_tempo(part, target_bpm: int):
    """Alle Tempo-Markierungen entfernen und neues Tempo setzen."""
    try:
        for el in part.recurse():
            if 'MetronomeMark' in el.classes:
                el.activeSite.remove(el)
    except Exception:
        pass

    part.insert(0, music21.tempo.MetronomeMark(number=target_bpm))


def _get_interactive_page_turns(score, frames_per_beat: float) -> list[int]:
    """Fragt interaktiv nach Seitenumbrüchen (Taktnummern).

    Wiederverwendet das Muster aus musescore_to_chroma.py.
    """
    print("\n--- SEITEN-LAYOUT ---")
    print("Gib die LETZTE Taktnummer jeder Seite ein (oder 'x' zum Beenden).\n")

    part = score.parts[0]
    page_indices = []
    page_counter = 1

    while True:
        try:
            user_input = input(f"  Letzte Taktnummer auf Seite {page_counter}? ")
            if user_input.strip().lower() == 'x':
                break

            measure_num = int(user_input)
            found = False

            for m in part.recurse():
                if 'Measure' in m.classes and m.number == measure_num:
                    abs_offset = m.getOffsetInHierarchy(score)
                    end_beat = abs_offset + m.duration.quarterLength
                    frame_index = int(end_beat * frames_per_beat)

                    page_indices.append(frame_index)
                    print(f"    -> Seite {page_counter} endet bei Takt {measure_num} (Frame {frame_index})")
                    page_counter += 1
                    found = True
                    break

            if not found:
                print(f"    Takt {measure_num} nicht gefunden!")

        except ValueError:
            print("    Bitte eine Zahl eingeben.")

    print(f"  {len(page_indices)} Seitenumbrüche konfiguriert.\n")
    return page_indices
