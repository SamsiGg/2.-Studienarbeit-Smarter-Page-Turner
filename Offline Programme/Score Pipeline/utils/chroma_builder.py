# =============================================================================
# chroma_builder.py – Chroma-Vektoren via Audio-Synthese berechnen
# =============================================================================
# Pipeline: MusicXML → MIDI → FluidSynth (Audio) → librosa (Chroma)
#
# Benötigt:
#   brew install fluidsynth
#   Soundfont in data/soundfonts/MuseScore_General.sf2
# =============================================================================

import os
import subprocess
import tempfile
import numpy as np
import music21
import librosa
import soundfile as sf

# Audio-Parameter (müssen mit Live-System übereinstimmen)
SAMPLE_RATE = 44100
FFT_SIZE = 4096
HOP_LENGTH = 512

# Schwellwert für stille Frames (Pausen): untere X% der RMS-Werte → Rauschen
RMS_SILENCE_PERCENTILE = 3

# Soundfont-Pfad (relativ zum Projekt)
SOUNDFONT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'soundfonts', 'MuseScore_General.sf2'
)

# MIDI-Programme für verschiedene Instrumente
INSTRUMENTS = {
    "piano": 0, "violin": 40, "violin_pizz": 45, "viola": 41, "cello": 42,
    "contrabass": 43, "flute": 73, "clarinet": 71, "trumpet": 56, "guitar": 24
}

# FluidSynth muss im PATH sein (macOS Homebrew)
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"


def build_chroma(musicxml_path: str, bpm: int, instrument: str = "violin",
                 sample_rate: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH,
                 wav_output_path: str = None) -> tuple[np.ndarray, list[int]]:
    """MusicXML → MIDI → FluidSynth → Chroma-Matrix + Seitenumbruch-Indizes.

    Args:
        musicxml_path: Pfad zur .musicxml oder .mxl Datei.
        bpm: Tempo in BPM (überschreibt Tempo-Markierungen in der Datei).
        instrument: Instrument für MIDI-Synthese (z.B. "violin", "piano").
        sample_rate: Samplerate (muss mit Live-System übereinstimmen).
        hop_length: Hop-Length (muss mit Live-System übereinstimmen).
        wav_output_path: Optionaler Pfad zum Speichern der synthetisierten WAV.

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

    # Grace Notes entfernen (verursachen stuck notes in FluidSynth)
    _remove_grace_notes(part)

    # Tempo und Instrument erzwingen
    midi_program = INSTRUMENTS.get(instrument.lower(), 40)
    _set_tempo_and_instrument(part, score, bpm, midi_program)

    # Frames-per-Beat für Seitenumbrüche
    frames_per_beat = (60.0 / bpm) * (sample_rate / hop_length)

    # Interaktive Seitenumbrüche (VOR Audio-Synthese, damit man die Partitur noch sieht)
    page_indices = _get_interactive_page_turns(score, frames_per_beat)

    # MIDI exportieren → FluidSynth → WAV → Chroma
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = os.path.join(tmpdir, "score.mid")
        wav_path = os.path.join(tmpdir, "score.wav")

        # 1. MIDI exportieren
        print(f"  Exportiere MIDI ({bpm} BPM, {instrument})...")
        score.write('midi', fp=midi_path)

        # 2. FluidSynth: MIDI → WAV (mit eingebautem Reverb)
        print(f"  Synthetisiere Audio via FluidSynth...")
        sf_path = os.path.abspath(SOUNDFONT_PATH)
        if not os.path.exists(sf_path):
            raise FileNotFoundError(
                f"Soundfont nicht gefunden: {sf_path}\n"
                f"Bitte MuseScore_General.sf2 in data/soundfonts/ ablegen."
            )

        fs_cmd = [
            "fluidsynth", "-ni", "-g", "1.0",
            "-F", wav_path, "-r", str(sample_rate),
            sf_path, midi_path
        ]
        try:
            subprocess.run(fs_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise FileNotFoundError(
                "FluidSynth nicht gefunden! Installieren mit: brew install fluidsynth"
            )

        # 3. WAV laden
        print(f"  Berechne Chroma aus synthetisiertem Audio...")
        y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)

        # 4. WAV optional speichern
        if wav_output_path:
            sf.write(wav_output_path, y, sr)
            print(f"  WAV gespeichert: {wav_output_path}")

        chroma_raw = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=FFT_SIZE, hop_length=hop_length, n_chroma=12
        )

        # RMS-Energie pro Frame berechnen
        rms = librosa.feature.rms(y=y, frame_length=FFT_SIZE, hop_length=hop_length)[0]
        rms = rms[:chroma_raw.shape[1]]

    # Leise Frames (Pausen/Reverb-Ausklang) → gleichmäßig verteiltes Rauschen
    rms_threshold = np.percentile(rms, RMS_SILENCE_PERCENTILE)
    silent_frames = rms < rms_threshold
    chroma_raw[:, silent_frames] = np.random.uniform(0.0, 1.0, size=chroma_raw[:, silent_frames].shape)
    n_silent = np.sum(silent_frames)
    print(f"  {n_silent} stille Frames mit Rauschen ersetzt ({n_silent / len(rms) * 100:.1f}%)")

    # L2-Normalisierung pro Frame
    norms = np.linalg.norm(chroma_raw, axis=0, keepdims=True)
    norms[norms == 0] = 1
    chroma = chroma_raw / norms

    num_frames = chroma.shape[1]
    print(f"  {num_frames} Frames, {len(page_indices)} Seitengrenzen")

    return chroma, page_indices


def _remove_grace_notes(part):
    """Entfernt alle Grace Notes aus der Stimme.

    Grace Notes haben keine echte Dauer und können bei der MIDI-Synthese
    zu 'stuck notes' führen (Note-On ohne korrektes Note-Off).
    Für Chroma-Erkennung sind sie irrelevant.
    """
    grace_notes = []
    for el in part.recurse():
        if isinstance(el, music21.note.GeneralNote) and el.duration.isGrace:
            grace_notes.append(el)

    for gn in grace_notes:
        gn.activeSite.remove(gn)

    if grace_notes:
        print(f"  {len(grace_notes)} Grace Notes entfernt (verhindert stuck notes)")


def _set_tempo_and_instrument(part, score, target_bpm: int, midi_program: int):
    """Alle Tempo-Markierungen entfernen, neues Tempo und Instrument setzen."""
    # Alte Tempos entfernen
    try:
        for el in part.recurse():
            if 'MetronomeMark' in el.classes:
                el.activeSite.remove(el)
    except Exception:
        pass

    # Neues Tempo setzen
    new_tempo = music21.tempo.MetronomeMark(number=target_bpm)
    part.insert(0, new_tempo)

    # Auch im ersten Takt explizit (doppelt hält besser für MIDI-Export)
    first_measure = part.getElementsByClass('Measure').first()
    if first_measure:
        first_measure.insert(0, new_tempo)

    # Instrument setzen
    try:
        for el in part.recurse():
            if 'Instrument' in el.classes:
                el.activeSite.remove(el)
    except Exception:
        pass

    new_inst = music21.instrument.Instrument()
    new_inst.midiProgram = midi_program
    part.insert(0, new_inst)


def _get_interactive_page_turns(score, frames_per_beat: float) -> list[int]:
    """Fragt interaktiv nach Seitenumbrüchen (Taktnummern)."""
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
