# =============================================================================
# record_measures.py – Hochauflösende Aufnahme ausgewählter Takte
# =============================================================================
# Nutzung (aus WAV extrahieren):
#   python record_measures.py --wav live.wav --measures 10 11 12
#
# Nutzung (Live-Aufnahme per Mikrofon):
#   python record_measures.py --live --measures 10 11 12 --bpm 60
#
# Ausgabe:
#   - WAV-Dateien pro Takt in --outdir (Standard: data/audio/measures/)
#   - Hochauflösende Plots: Wellenform + Spektrogramm
# =============================================================================

import argparse
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- KONFIGURATION ---
DEFAULT_WAV = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/Fiocco_Musescore.wav"
DEFAULT_OUTDIR = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/audio/measures"

SAMPLE_RATE   = 44100   # Hz (hochauflösend)
HOP_LENGTH    = 256     # kleiner als in compare_scores.py → höhere Zeitauflösung
N_FFT         = 8192    # größer → höhere Frequenzauflösung
BPM           = 40
BEATS_PER_MEASURE = 4
AUFTAKT_BEATS = 0.5     # Achtel-Auftakt


# --- HELPER ---
def measure_to_time_range(measure: int, bpm: float, beats: int, auftakt_beats: float = 0.0):
    """Gibt Start- und Endzeit (Sekunden) für einen Takt zurück."""
    sec_per_beat    = 60.0 / bpm
    sec_per_measure = sec_per_beat * beats
    auftakt_sec     = auftakt_beats * sec_per_beat
    start_sec = (measure - 1) * sec_per_measure + auftakt_sec
    end_sec   =  measure      * sec_per_measure + auftakt_sec
    return start_sec, end_sec


def extract_measure_audio(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Schneidet ein Audio-Segment aus dem Array aus."""
    start_sample = int(start_sec * sr)
    end_sample   = int(end_sec   * sr)
    start_sample = max(0, start_sample)
    end_sample   = min(len(y), end_sample)
    return y[start_sample:end_sample]


def record_live(duration_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Nimmt Audio vom Standard-Mikrofon auf (benötigt sounddevice)."""
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("sounddevice nicht installiert. Bitte: pip install sounddevice")

    print(f"  Aufnahme startet in 3 s … ({duration_sec:.1f} s)")
    import time
    for i in range(3, 0, -1):
        print(f"  {i}…")
        time.sleep(1)
    print("  *** AUFNAHME LÄUFT ***")
    recording = sd.rec(int(duration_sec * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("  *** AUFNAHME BEENDET ***")
    return recording.flatten()


def plot_measure(y: np.ndarray, sr: int, measure: int, hop: int, n_fft: int,
                 save_path: str | None = None):
    """Zeigt Wellenform + hochauflösendes Mel-Spektrogramm + Chromagramm."""
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Takt {measure} – Hochauflösende Analyse", fontsize=15, fontweight='bold')

    gs = gridspec.GridSpec(3, 1, hspace=0.55)

    # --- 1) Wellenform ---
    ax1 = fig.add_subplot(gs[0])
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax1.plot(times, y, color='steelblue', linewidth=0.5)
    ax1.set_xlabel("Zeit (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Wellenform")
    ax1.set_xlim(0, len(y) / sr)

    # --- 2) Mel-Spektrogramm (hochauflösend) ---
    ax2 = fig.add_subplot(gs[1])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                        n_mels=256, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                                    x_axis='time', y_axis='mel',
                                    fmax=8000, cmap='magma', ax=ax2)
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    ax2.set_title("Mel-Spektrogramm (hochauflösend)")
    ax2.set_xlabel("Zeit (s)")

    # --- 3) Chromagramm ---
    ax3 = fig.add_subplot(gs[2])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                          tuning=0, n_chroma=12)
    img2 = librosa.display.specshow(chroma, sr=sr, hop_length=hop,
                                     x_axis='time', y_axis='chroma',
                                     cmap='coolwarm', ax=ax3, vmin=0, vmax=1)
    fig.colorbar(img2, ax=ax3)
    ax3.set_title("Chromagramm")
    ax3.set_xlabel("Zeit (s)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot gespeichert: {save_path}")
    plt.show()
    plt.close(fig)


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(
        description="Hochauflösende Aufnahme / Extraktion ausgewählter Takte."
    )
    parser.add_argument("--wav",      default=DEFAULT_WAV,
                        help="Quell-WAV-Datei (wird ignoriert wenn --live)")
    parser.add_argument("--live",     action="store_true",
                        help="Live-Aufnahme per Mikrofon statt WAV-Datei")
    parser.add_argument("--measures", nargs='+', type=int, default=[20, 21, 22],
                        help="Taktnummern (z.B. --measures 10 11 12)")
    parser.add_argument("--bpm",      type=float, default=BPM,
                        help="Tempo in BPM")
    parser.add_argument("--beats",    type=int,   default=BEATS_PER_MEASURE,
                        help="Schläge pro Takt (Standard: 4)")
    parser.add_argument("--auftakt",  type=float, default=AUFTAKT_BEATS,
                        help="Auftakt in Viertelnoten (Standard: 0.5)")
    parser.add_argument("--sr",       type=int,   default=SAMPLE_RATE,
                        help="Sample-Rate Hz (Standard: 44100)")
    parser.add_argument("--outdir",   default=DEFAULT_OUTDIR,
                        help="Ausgabe-Ordner für WAV-Dateien und Plots")
    parser.add_argument("--no-save",  action="store_true",
                        help="Kein Speichern von WAV/Plots – nur anzeigen")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- AUDIO LADEN / AUFNEHMEN ---
    if args.live:
        sec_per_beat    = 60.0 / args.bpm
        sec_per_measure = sec_per_beat * args.beats
        # Alle Takte zusammen aufnehmen (von erstem bis letztem + Puffer)
        first_measure = min(args.measures)
        last_measure  = max(args.measures)
        _, end_last = measure_to_time_range(last_measure, args.bpm, args.beats, args.auftakt)
        start_first, _ = measure_to_time_range(first_measure, args.bpm, args.beats, args.auftakt)
        duration = end_last + 1.0  # 1 s Puffer
        print(f"Live-Aufnahme: {duration:.1f} s (Takte {first_measure}–{last_measure})")
        y_full = record_live(duration, sr=args.sr)
        source_label = "live"
    else:
        print(f"Lade Audio: {args.wav}")
        y_full, _ = librosa.load(args.wav, sr=args.sr, mono=True)
        source_label = os.path.splitext(os.path.basename(args.wav))[0]

    print(f"Audio geladen: {len(y_full)/args.sr:.1f} s bei {args.sr} Hz\n")

    # --- PRO TAKT VERARBEITEN ---
    for measure in args.measures:
        start_sec, end_sec = measure_to_time_range(measure, args.bpm, args.beats, args.auftakt)
        y_measure = extract_measure_audio(y_full, args.sr, start_sec, end_sec)

        duration_actual = len(y_measure) / args.sr
        print(f"Takt {measure:3d}  ({start_sec:.2f} s – {end_sec:.2f} s, "
              f"{duration_actual:.2f} s, {len(y_measure)} Samples)")

        if not args.no_save:
            wav_out  = os.path.join(args.outdir, f"{source_label}_takt{measure:03d}.wav")
            plot_out = os.path.join(args.outdir, f"{source_label}_takt{measure:03d}.png")
            sf.write(wav_out, y_measure, args.sr, subtype='PCM_24')
            print(f"  WAV gespeichert:  {wav_out}")
        else:
            plot_out = None

        plot_measure(y_measure, args.sr, measure, HOP_LENGTH, N_FFT,
                     save_path=plot_out if not args.no_save else None)

    print("\nFertig.")


if __name__ == "__main__":
    main()
