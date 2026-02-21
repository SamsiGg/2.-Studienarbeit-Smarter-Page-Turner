# =============================================================================
# compare_scores.py – Partitur-Chroma vs Live-Audio-Chroma vergleichen
# =============================================================================
# Nutzung:
#   python compare_scores.py                           (Standard-Pfade)
#   python compare_scores.py --score Fiocco.npz --wav live.wav
#   python compare_scores.py --measures 10 20 30 40 50
# =============================================================================

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt

# --- KONFIGURATION ---
SCORE_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/Pachelbel_Musescore.npz"
LIVE_WAV_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/audio/Pachelbel-Live-35bpm.wav"

SAMPLE_RATE = 44100
HOP_LENGTH = 512
BPM = 35
BEATS_PER_MEASURE = 4

# Welche Takte vergleichen? (5 Stück)
COMPARE_MEASURES = [18, 19, 20, 21, 22]
#COMPARE_MEASURES = [20, 21, 22, 23, 24]

CHROMA_LABELS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --- HELPER ---
def measure_to_frame_range(measure, sr, hop, bpm, beats, auftakt_beats=0.0):
    """Gibt Start- und End-Frame für einen Takt zurück.

    auftakt_beats: Länge des Auftakts in Vierteln (z.B. 0.5 für eine Achtel).
    """
    sec_per_beat = 60.0 / bpm
    sec_per_measure = sec_per_beat * beats
    sec_per_frame = hop / sr

    # Auftakt verschiebt alles nach vorne
    auftakt_sec = auftakt_beats * sec_per_beat
    start_sec = (measure - 1) * sec_per_measure + auftakt_sec
    end_sec = measure * sec_per_measure + auftakt_sec

    start_frame = int(start_sec / sec_per_frame)
    end_frame = int(end_sec / sec_per_frame)
    return start_frame, end_frame

def load_score_chroma(filepath):
    """Lädt Chroma aus .npz oder .h Datei. Gibt (12, N) zurück."""
    if filepath.endswith('.npz'):
        archive = np.load(filepath)
        return archive['chroma']  # (12, N)
    else:
        from dtw_engine import load_h_file_chroma
        return load_h_file_chroma(filepath)


def main():
    parser = argparse.ArgumentParser(description="Vergleicht Partitur-Chroma mit Live-Audio-Chroma.")
    parser.add_argument("--score", default=SCORE_FILE, help="Partitur-Datei (.npz oder .h)")
    parser.add_argument("--wav", default=LIVE_WAV_FILE, help="Live-Audio WAV-Datei")
    parser.add_argument("--measures", nargs='+', type=int, default=COMPARE_MEASURES,
                        help="Taktnummern zum Vergleichen (z.B. --measures 10 20 30)")
    parser.add_argument("--bpm", type=int, default=BPM, help="Tempo in BPM")
    parser.add_argument("--auftakt", type=float, default=0.5,
                        help="Auftakt-Länge in Viertelnoten (0.5 = Achtel, 1.0 = Viertel, Standard: 0.5)")
    args = parser.parse_args()

    # --- DATEN LADEN ---
    print(f"Lade Partitur: {args.score}...")
    score_chroma = load_score_chroma(args.score)  # (12, N)

    print(f"Lade Live-Audio: {args.wav}...")
    y, sr = librosa.load(args.wav, sr=SAMPLE_RATE, mono=True)
    live_chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=4096, hop_length=HOP_LENGTH, tuning=0, n_chroma=12
    )  # (12, M)

    # L2-Normalisierung
    norm_s = np.linalg.norm(score_chroma, axis=0, keepdims=True)
    norm_s[norm_s == 0] = 1
    score_norm = score_chroma / norm_s

    norm_l = np.linalg.norm(live_chroma, axis=0, keepdims=True)
    norm_l[norm_l == 0] = 1
    live_norm = live_chroma / norm_l

    print(f"Partitur: {score_chroma.shape[1]} Frames")
    print(f"Live:     {live_chroma.shape[1]} Frames")

    # --- PLOT: Chromagramme pro Takt ---
    compare_measures = args.measures
    n_measures = len(compare_measures)
    fig, axes = plt.subplots(n_measures, 2, figsize=(14, 3 * n_measures))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    for row, measure in enumerate(compare_measures):
        start_f, end_f = measure_to_frame_range(measure, SAMPLE_RATE, HOP_LENGTH, args.bpm, BEATS_PER_MEASURE, args.auftakt)

        # Sicherstellen, dass wir nicht über die Grenzen gehen
        end_f_score = min(end_f, score_chroma.shape[1])
        end_f_live = min(end_f, live_chroma.shape[1])

        score_slice = score_norm[:, start_f:end_f_score]
        live_slice = live_norm[:, start_f:end_f_live]

        # Durchschnittlicher Chroma-Vektor pro Takt
        score_avg = np.mean(score_slice, axis=1) if score_slice.shape[1] > 0 else np.zeros(12)
        live_avg = np.mean(live_slice, axis=1) if live_slice.shape[1] > 0 else np.zeros(12)

        # Kosinus-Distanz für diesen Takt
        dot = np.dot(score_avg / (np.linalg.norm(score_avg) + 1e-9),
                     live_avg / (np.linalg.norm(live_avg) + 1e-9))
        cos_dist = 1.0 - max(-1, min(1, dot))

        # Balkendiagramm: Partitur vs Live
        x = np.arange(12)
        width = 0.35

        axes[row, 0].bar(x - width/2, score_avg, width, label='Partitur', color='steelblue')
        axes[row, 0].bar(x + width/2, live_avg, width, label='Live', color='coral')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels(CHROMA_LABELS, fontsize=8)
        axes[row, 0].set_title(f"Takt {measure} – Chroma (cos_dist={cos_dist:.3f})")
        axes[row, 0].set_ylim(0, 1)
        axes[row, 0].legend(fontsize=7)

        # Chromagramm über Zeit (Heatmap)
        max_frames = max(score_slice.shape[1], live_slice.shape[1], 1)
        combined = np.zeros((24, max_frames))
        combined[:12, :score_slice.shape[1]] = score_slice
        combined[12:, :live_slice.shape[1]] = live_slice

        axes[row, 1].imshow(combined, aspect='auto', origin='lower', cmap='magma',
                            vmin=0, vmax=1)
        axes[row, 1].axhline(y=11.5, color='white', linewidth=2, linestyle='--')
        axes[row, 1].set_yticks([5, 17])
        axes[row, 1].set_yticklabels(['Partitur', 'Live'], fontsize=9)
        axes[row, 1].set_title(f"Takt {measure} – Zeitverlauf")
        axes[row, 1].set_xlabel("Frame im Takt")

    plt.suptitle("Partitur vs Live-Audio – Chroma-Vergleich pro Takt", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
