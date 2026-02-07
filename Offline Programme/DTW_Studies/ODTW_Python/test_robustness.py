"""
ODTW Robustness Testing
-----------------------
Testet die Robustheit des Online-DTW-Algorithmus gegen:
- Tempo-Variationen (Speed Factors)
- Audio-Rauschen (Noise)
- Verschiedene Input-Formate (.wav Audio oder .npy Chroma)

Author: Samuel Geffert
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from collections import deque
from dtw_engine import StandardODTW, DebugODTW, load_h_file_chroma

# --- KONFIGURATION ---
SCORE_FILE = "data/ScoreData.h"
LIVE_NPY_FILE = "data/Fiocco-Live (40bpm)_chroma.npy"
LIVE_WAV_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/MusescoreToChroma/Fiocco.wav"

# Musikalische Parameter
SAMPLE_RATE = 44100
HOP_LENGTH = 512
BPM = 40
BEATS_PER_MEASURE = 4

# --- HELPER FUNCTIONS ---

def frames_to_measures(n_frames):
    """Rechnet Frames in Takte um (basierend auf BPM und Hop-Size)"""
    sec_per_frame = HOP_LENGTH / SAMPLE_RATE
    total_seconds = n_frames * sec_per_frame
    sec_per_measure = (60.0 / BPM) * BEATS_PER_MEASURE
    return total_seconds / sec_per_measure

def stretch_chroma(chroma, speed_factor):
    """
    Verändert die Geschwindigkeit der Chroma-Matrix.
    speed_factor > 1.0: Schneller (Matrix wird kürzer)
    speed_factor < 1.0: Langsamer (Matrix wird länger)
    """
    zoom_factor = 1 / speed_factor
    return zoom(chroma, (1, zoom_factor), order=1)

def load_chroma_from_wav(wav_file):
    """Lädt Audio und extrahiert Chroma-Features"""
    print(f"Lade WAV: {wav_file}...")
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=4096, hop_length=HOP_LENGTH, tuning=0
    )
    return chroma

def load_chroma_from_npy(npy_file):
    """Lädt vorberechnete Chroma-Daten aus .npy"""
    print(f"Lade NPY: {npy_file}...")
    chroma = np.load(npy_file)
    # Falls Shape (N, 12) ist, transponieren zu (12, N)
    if chroma.shape[0] > 12 and chroma.shape[1] == 12:
        chroma = chroma.T
    return chroma

def prepare_simulation_data(base_chroma, speed_factor, noise_level):
    """
    Verändert Tempo und fügt Rauschen hinzu.
    speed_factor > 1.0 = Schneller spielen (kürzeres Array)
    """
    # 1. Time Stretching
    if speed_factor != 1.0:
        live_chroma = stretch_chroma(base_chroma, speed_factor)
    else:
        live_chroma = base_chroma.copy()

    # 2. Noise Injection
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, live_chroma.shape)
        live_chroma = live_chroma + noise
        live_chroma = np.clip(live_chroma, 0, None)

    # 3. Neu normalisieren
    norms = np.linalg.norm(live_chroma, axis=0)
    norms[norms == 0] = 1
    live_chroma = live_chroma / norms

    return live_chroma

# --- SIMULATION FUNCTIONS ---

def run_simulation_standard(ref_chroma, live_base, speed, noise, label):
    """Führt Simulation mit StandardODTW durch"""
    print(f"\n--- {label}: Tempo {speed*100:.0f}%, Noise {noise} ---")

    # Daten vorbereiten
    live_input = prepare_simulation_data(live_base, speed, noise)

    # ODTW Engine
    dtw = StandardODTW(ref_chroma)

    positions = []
    costs = []
    n_frames = live_input.shape[1]

    # Simulation Loop
    for i in range(n_frames):
        live_vec = live_input[:, i]
        pos, cost = dtw.step(live_vec)
        positions.append(pos)
        costs.append(cost)

        if i % 300 == 0 or i == n_frames - 1:
            progress = (i / n_frames) * 100
            print(f"\rVerarbeite {n_frames} Frames... [{progress:5.1f}%]", end="")

    print()
    return positions, costs, n_frames

def run_simulation_debug(ref_chroma, live_base, speed, noise, label):
    """Führt Simulation mit DebugODTW durch (gibt mehr Informationen zurück)"""
    print(f"\n--- {label}: Tempo {speed*100:.0f}%, Noise {noise} ---")

    live_input = prepare_simulation_data(live_base, speed, noise)
    dtw = DebugODTW(ref_chroma)

    history_pos = []
    history_global = []
    history_local = []

    for frame in live_input.T:
        pos, g_cost, l_cost = dtw.step(frame)
        history_pos.append(pos)
        history_global.append(g_cost)
        history_local.append(l_cost)

    return history_pos, history_global, history_local, len(live_input.T)

# --- PLOTTING FUNCTIONS ---

def plot_tracking_comparison(ref_chroma, live_base, input_type="npy"):
    """
    Plottet Tracking-Pfade für verschiedene Tempo/Noise-Kombinationen
    """
    ref_frames = ref_chroma.shape[1]
    total_measures_score = frames_to_measures(ref_frames)

    print(f"\nLänge der Partitur: {ref_frames} Frames = {total_measures_score:.2f} Takte")

    plt.figure(figsize=(10, 8))

    # Szenarien
    scenarios = [
        (1.0, 0.0, 'green', 'Normal (1.0x, kein Noise)', '-'),
        (1.2, 0.2, 'red', 'Schnell (1.2x) + Noise', '--'),
        (0.8, 0.2, 'blue', 'Langsam (0.8x) + Noise', ':'),
    ]

    for speed, noise, color, label, style in scenarios:
        positions, _, n_input = run_simulation_standard(
            ref_chroma, live_base, speed, noise, label
        )

        # X-Achse: Normiert auf Taktlänge
        x_axis = np.linspace(0, total_measures_score, n_input)

        # Y-Achse: Erkannte Position in Takten
        y_axis = [frames_to_measures(p) for p in positions]

        plt.plot(x_axis, y_axis, label=label, color=color,
                linestyle=style, linewidth=2)

    # Ideale Diagonale
    plt.plot([0, total_measures_score], [0, total_measures_score],
            'k-', alpha=0.3, label="Ideal")

    plt.title(f"ODTW Tracking Robustheit ({input_type.upper()} Input)\nBPM: {BPM}")
    plt.xlabel("Fortschritt im Input (Takte)")
    plt.ylabel("Erkannte Position (Takte in Partitur)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_cost_analysis(ref_chroma, live_base):
    """
    Plottet detaillierte Kosten-Analyse mit drei Subplots:
    1. Tracking Position
    2. Globale Kosten (Raw)
    3. Globale Kosten (Moving Average)
    """
    scenarios = [
        (1.0, 0.05, 'green', 'Normal (1.0x, wenig Noise)'),
        (1.3, 0.3, 'red', 'Schnell (1.3x, viel Noise)'),
        (0.7, 0.3, 'blue', 'Langsam (0.7x, viel Noise)')
    ]

    fig, (ax_path, ax_global, ax_avg) = plt.subplots(3, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)

    ref_len = ref_chroma.shape[1]
    WINDOW_SIZE = 300

    for speed, noise, color, label in scenarios:
        pos, glob, loc, input_len = run_simulation_debug(
            ref_chroma, live_base, speed, noise, label
        )

        x_axis = np.linspace(0, 100, input_len)

        # Plot 1: Tracking Pfad
        ax_path.plot(x_axis, pos, color=color, label=label, linewidth=1.5)

        # Plot 2: Globale Kosten (Raw)
        ax_global.plot(x_axis, glob, color=color, label=label, linewidth=1.5)

        # Plot 3: Moving Average
        if len(glob) >= WINDOW_SIZE:
            glob_smooth = np.convolve(glob, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
            x_axis_smooth = np.linspace(0, 100, len(glob_smooth))
            ax_avg.plot(x_axis_smooth, glob_smooth, color=color, label=label, linewidth=2)

    # Formatierung
    ax_path.set_title("1. Tracking Position")
    ax_path.set_ylabel("Frame in Partitur")
    ax_path.grid(True)
    ax_path.plot([0, 100], [0, ref_len], 'k--', alpha=0.3, label='Ideal')
    ax_path.legend(loc='upper left')

    ax_global.set_title("2. Globale Kosten (Rohwerte)")
    ax_global.set_ylabel("Akkumulierte Kosten")
    ax_global.grid(True)

    ax_avg.set_title(f"3. Globale Kosten (Moving Average über {WINDOW_SIZE} Frames)")
    ax_avg.set_ylabel("Ø Kosten")
    ax_avg.set_xlabel("Fortschritt im Input (%)")
    ax_avg.grid(True)
    ax_avg.axhline(y=10.0, color='black', linestyle='--', alpha=0.5,
                   label='Mögliche Lost-Schwelle?')
    ax_avg.legend(loc='upper left')

    plt.suptitle("ODTW Kosten-Analyse (Damping=0.96)", fontsize=16)
    plt.show()

# --- MAIN ---

def main():
    """Hauptprogramm - wähle Input-Typ und Plot-Typ"""

    # Referenz-Chroma laden
    try:
        ref_chroma = load_h_file_chroma(SCORE_FILE)
    except Exception as e:
        print(f"Fehler beim Laden der Referenz: {e}")
        return

    # Wähle Input-Typ
    print("\n=== ODTW Robustness Testing ===")
    print("Wähle Input-Typ:")
    print("  [1] NPY (vorberechnete Chroma)")
    print("  [2] WAV (Audio-Datei)")

    choice = input("Eingabe (1 oder 2): ").strip()

    if choice == "1":
        try:
            live_base = load_chroma_from_npy(LIVE_NPY_FILE)
            input_type = "npy"
        except FileNotFoundError:
            print(f"Fehler: {LIVE_NPY_FILE} nicht gefunden!")
            return
    elif choice == "2":
        try:
            live_base = load_chroma_from_wav(LIVE_WAV_FILE)
            input_type = "wav"
        except FileNotFoundError:
            print(f"Fehler: {LIVE_WAV_FILE} nicht gefunden!")
            return
    else:
        print("Ungültige Eingabe.")
        return

    # Wähle Plot-Typ
    print("\nWähle Analyse-Typ:")
    print("  [1] Tracking Comparison (einfach)")
    print("  [2] Kosten-Analyse (detailliert)")

    plot_choice = input("Eingabe (1 oder 2): ").strip()

    if plot_choice == "1":
        plot_tracking_comparison(ref_chroma, live_base, input_type)
    elif plot_choice == "2":
        plot_cost_analysis(ref_chroma, live_base)
    else:
        print("Ungültige Eingabe.")

if __name__ == "__main__":
    main()
