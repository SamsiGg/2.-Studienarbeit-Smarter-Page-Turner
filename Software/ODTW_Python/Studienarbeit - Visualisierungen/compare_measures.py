# =============================================================================
# compare_measures.py – Hochauflösender Takt-Vergleich zweier Audio-Quellen
# =============================================================================
# Nutzung:
#   python compare_measures.py                              (Standard-Pfade)
#   python compare_measures.py --wav-a ref.wav --wav-b live.wav
#   python compare_measures.py --measures 10 11 12 --bpm 40
# =============================================================================

import argparse
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- KONFIGURATION ---
DEFAULT_WAV_A = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/Fiocco_Musescore.wav"
DEFAULT_WAV_B = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/audio/Fiocco-Live-40bpm.wav"

SAMPLE_RATE       = 44100
HOP_LENGTH        = 256
N_FFT             = 8192
BPM               = 40
BEATS_PER_MEASURE = 4
AUFTAKT_BEATS     = 0.5

COMPARE_MEASURES  = [20, 21, 22]

CHROMA_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


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
    start_sample = max(0, int(start_sec * sr))
    end_sample   = min(len(y), int(end_sec * sr))
    return y[start_sample:end_sample]


def compute_features(y: np.ndarray, sr: int, hop: int, n_fft: int):
    """Berechnet Mel-Spektrogramm (dB) und normalisiertes Chromagramm."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                        n_mels=256, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                          tuning=0, n_chroma=12)
    norm = np.linalg.norm(chroma, axis=0, keepdims=True)
    norm[norm == 0] = 1
    chroma_norm = chroma / norm

    return S_db, chroma_norm


def cosine_distance(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """Kosinus-Distanz zwischen den mittleren Chroma-Vektoren zweier Takte."""
    avg_a = np.mean(chroma_a, axis=1) if chroma_a.shape[1] > 0 else np.zeros(12)
    avg_b = np.mean(chroma_b, axis=1) if chroma_b.shape[1] > 0 else np.zeros(12)
    dot = np.dot(avg_a / (np.linalg.norm(avg_a) + 1e-9),
                 avg_b / (np.linalg.norm(avg_b) + 1e-9))
    return 1.0 - float(np.clip(dot, -1, 1))


def plot_measure_comparison(measure: int,
                             y_a: np.ndarray, y_b: np.ndarray, sr: int,
                             hop: int, n_fft: int,
                             label_a: str, label_b: str,
                             save_path: str | None = None):
    """
    Vergleichs-Plot für einen Takt:
      Zeile 1 – Wellenform A | Wellenform B
      Zeile 2 – Mel-Spektrogramm A | Mel-Spektrogramm B
      Zeile 3 – Chromagramm A | Chromagramm B
      Zeile 4 – Chroma-Balkendiagramm (Mittelwert A vs B nebeneinander)
    """
    S_db_a, chroma_a = compute_features(y_a, sr, hop, n_fft)
    S_db_b, chroma_b = compute_features(y_b, sr, hop, n_fft)
    cos_dist = cosine_distance(chroma_a, chroma_b)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Takt {measure} – Vergleich: {label_a}  vs.  {label_b}\n"
        f"Kosinus-Distanz (Chroma): {cos_dist:.4f}",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(4, 2, hspace=0.65, wspace=0.35)

    colors = {'a': 'steelblue', 'b': 'coral'}

    # ── Zeile 1: Wellenform ──────────────────────────────────────────────────
    for col, (y, label, color) in enumerate([(y_a, label_a, colors['a']),
                                              (y_b, label_b, colors['b'])]):
        ax = fig.add_subplot(gs[0, col])
        times = np.linspace(0, len(y) / sr, num=len(y))
        ax.plot(times, y, color=color, linewidth=0.4)
        ax.set_title(f"Wellenform – {label}")
        ax.set_xlabel("Zeit (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, len(y) / sr)

    # ── Zeile 2: Mel-Spektrogramm ────────────────────────────────────────────
    # Gemeinsame vmin/vmax für direkten Vergleich
    vmin = min(S_db_a.min(), S_db_b.min())
    vmax = max(S_db_a.max(), S_db_b.max())

    for col, (S_db, label) in enumerate([(S_db_a, label_a), (S_db_b, label_b)]):
        ax = fig.add_subplot(gs[1, col])
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                                        x_axis='time', y_axis='mel',
                                        fmax=8000, cmap='magma', ax=ax,
                                        vmin=vmin, vmax=vmax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f"Mel-Spektrogramm – {label}")
        ax.set_xlabel("Zeit (s)")

    # ── Zeile 3: Chromagramm ─────────────────────────────────────────────────
    for col, (chroma, label) in enumerate([(chroma_a, label_a), (chroma_b, label_b)]):
        ax = fig.add_subplot(gs[2, col])
        img2 = librosa.display.specshow(chroma, sr=sr, hop_length=hop,
                                         x_axis='time', y_axis='chroma',
                                         cmap='coolwarm', ax=ax, vmin=0, vmax=1)
        fig.colorbar(img2, ax=ax)
        ax.set_title(f"Chromagramm – {label}")
        ax.set_xlabel("Zeit (s)")

    # ── Zeile 4: Chroma-Mittelwert-Balken (A vs B) ───────────────────────────
    avg_a = np.mean(chroma_a, axis=1) if chroma_a.shape[1] > 0 else np.zeros(12)
    avg_b = np.mean(chroma_b, axis=1) if chroma_b.shape[1] > 0 else np.zeros(12)

    ax_bar = fig.add_subplot(gs[3, :])   # volle Breite
    x = np.arange(12)
    w = 0.35
    ax_bar.bar(x - w/2, avg_a, w, label=label_a, color=colors['a'])
    ax_bar.bar(x + w/2, avg_b, w, label=label_b, color=colors['b'])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CHROMA_LABELS)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Chroma (normiert)")
    ax_bar.set_title(f"Mittlerer Chroma-Vektor pro Takt  (cos_dist = {cos_dist:.4f})")
    ax_bar.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot gespeichert: {save_path}")
    plt.show()
    plt.close(fig)


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(
        description="Hochauflösender Takt-Vergleich zweier Audio-Quellen."
    )
    parser.add_argument("--wav-a",    default=DEFAULT_WAV_A,
                        help="Erste Audio-Quelle (Referenz)")
    parser.add_argument("--wav-b",    default=DEFAULT_WAV_B,
                        help="Zweite Audio-Quelle (z.B. Live-Aufnahme)")
    parser.add_argument("--label-a",  default=None,
                        help="Bezeichnung für Quelle A (Standard: Dateiname)")
    parser.add_argument("--label-b",  default=None,
                        help="Bezeichnung für Quelle B (Standard: Dateiname)")
    parser.add_argument("--measures", nargs='+', type=int, default=COMPARE_MEASURES,
                        help="Taktnummern (z.B. --measures 10 11 12)")
    parser.add_argument("--bpm",      type=float, default=BPM,
                        help="Tempo in BPM")
    parser.add_argument("--beats",    type=int,   default=BEATS_PER_MEASURE,
                        help="Schläge pro Takt (Standard: 4)")
    parser.add_argument("--auftakt",  type=float, default=AUFTAKT_BEATS,
                        help="Auftakt in Viertelnoten (Standard: 0.5)")
    parser.add_argument("--sr",       type=int,   default=SAMPLE_RATE,
                        help="Sample-Rate Hz (Standard: 44100)")
    parser.add_argument("--outdir",   default=None,
                        help="Ausgabe-Ordner für Plots (kein Speichern wenn weggelassen)")
    args = parser.parse_args()

    label_a = args.label_a or os.path.splitext(os.path.basename(args.wav_a))[0]
    label_b = args.label_b or os.path.splitext(os.path.basename(args.wav_b))[0]

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    # --- BEIDE QUELLEN LADEN ---
    print(f"Lade A: {args.wav_a}")
    y_a, _ = librosa.load(args.wav_a, sr=args.sr, mono=True)
    print(f"Lade B: {args.wav_b}")
    y_b, _ = librosa.load(args.wav_b, sr=args.sr, mono=True)
    print(f"A: {len(y_a)/args.sr:.1f} s   B: {len(y_b)/args.sr:.1f} s\n")

    # --- PRO TAKT VERGLEICHEN ---
    for measure in args.measures:
        start_sec, end_sec = measure_to_time_range(
            measure, args.bpm, args.beats, args.auftakt
        )
        y_m_a = extract_measure_audio(y_a, args.sr, start_sec, end_sec)
        y_m_b = extract_measure_audio(y_b, args.sr, start_sec, end_sec)

        _, chroma_a = compute_features(y_m_a, args.sr, HOP_LENGTH, N_FFT)
        _, chroma_b = compute_features(y_m_b, args.sr, HOP_LENGTH, N_FFT)
        cos_dist = cosine_distance(chroma_a, chroma_b)

        print(f"Takt {measure:3d}  ({start_sec:.2f} s – {end_sec:.2f} s)  "
              f"cos_dist = {cos_dist:.4f}")

        save_path = None
        if args.outdir:
            save_path = os.path.join(
                args.outdir, f"compare_takt{measure:03d}_{label_a}_vs_{label_b}.png"
            )

        plot_measure_comparison(
            measure, y_m_a, y_m_b, args.sr, HOP_LENGTH, N_FFT,
            label_a, label_b, save_path=save_path
        )

    print("\nFertig.")


if __name__ == "__main__":
    main()
