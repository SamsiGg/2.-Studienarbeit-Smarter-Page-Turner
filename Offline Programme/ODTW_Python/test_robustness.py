"""
ODTW Robustness Testing
-----------------------
Testet die Robustheit des Online-DTW-Algorithmus gegen:
- Tempo-Variationen (Speed Factors)
- Audio-Rauschen (Noise)

Features:
- Lädt Audio von WAV (wie im echten Teensy)
- Automatisches Speichern von .npy Dateien für Verifikation
- Debug-Analyse mit Tracking Position und Globalen Kosten

Bsp: python3 test_robustness.py --recovery --jump 50 30

Author: Samuel Geffert
"""

import os
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from collections import deque
from dtw_engine import StandardODTW, DebugODTW, load_score_chroma
from recovery_odtw import RecoveryODTW

# --- KONFIGURATION ---
SCORE_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/Pachelbel_Musescore.npz"
LIVE_WAV_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/audio/Pachelbel-Live-35bpm.wav"
NPY_OUTPUT_DIR = "../data/generated"  # NPY-Dateien werden hier gespeichert

# Musikalische Parameter
SAMPLE_RATE = 44100
HOP_LENGTH = 512
BPM = 40
BEATS_PER_MEASURE = 4

# Test-Szenarien (Tempo & Noise)
TEST_SCENARIOS = [
    (1.0, 0.05, 'green', 'Normal (1.0x, wenig Noise)'),
    (1.3, 0.3, 'red', 'Schnell (1.3x, viel Noise)'),
    (0.7, 0.3, 'blue', 'Langsam (0.7x, viel Noise)')
]
# Format: (speed_factor, noise_level, color, label)
# speed_factor: 1.0 = normal, >1.0 = schneller, <1.0 = langsamer
# noise_level: 0.0 = kein Noise, 0.3 = viel Noise

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

def load_chroma_from_wav(wav_file, save_npy=True):
    """
    Lädt Audio und extrahiert Chroma-Features.
    Speichert optional als .npy für spätere Verwendung/Verifikation.
    """
    print(f"Lade WAV: {wav_file}...")

    # Audio laden
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)

    # Chroma berechnen
    chroma_raw = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=4096, hop_length=HOP_LENGTH, tuning=0, n_chroma=12
    )

    # Optional: Als .npy speichern (für Verifikation mit audio_generator)
    if save_npy:
        # Transponieren und L2-Normalisieren (wie AudioToChroma.py)
        chroma_steps = chroma_raw.T  # [12, Zeit] → [Zeit, 12]
        norms = np.linalg.norm(chroma_steps, axis=1, keepdims=True)
        chroma_l2 = chroma_steps / (norms + 1e-9)

        # Speichern in data/generated/
        base_name = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(base_name)[0]
        npy_file = os.path.join(NPY_OUTPUT_DIR, name_without_ext + "_chroma.npy")

        # Stelle sicher, dass der Ordner existiert
        os.makedirs(NPY_OUTPUT_DIR, exist_ok=True)

        np.save(npy_file, chroma_l2)
        print(f"💾 Chroma als .npy gespeichert: {npy_file}")

    return chroma_raw

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

def inject_jump_back(chroma, jump_from_measure, jump_to_measure, total_measures):
    """Simuliert einen Rückwärtssprung (z.B. Stelle nochmal üben).

    Args:
        chroma: (12, N) Chroma-Matrix
        jump_from_measure: Takt ab dem gesprungen wird
        jump_to_measure: Ziel-Takt wohin gesprungen wird
        total_measures: Gesamtanzahl Takte (für Frame-Berechnung)

    Returns:
        Neue Chroma-Matrix mit dem Sprung eingebaut
    """
    n = chroma.shape[1]
    jump_from = int(n * jump_from_measure / total_measures)
    jump_to = int(n * jump_to_measure / total_measures)

    part1 = chroma[:, :jump_from]
    part2 = chroma[:, jump_to:]

    result = np.hstack([part1, part2])
    print(f"  Sprung eingebaut: Takt {jump_from_measure} -> Takt {jump_to_measure} "
          f"(Frame {jump_from} -> {jump_to})")
    return result

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

def run_simulation_recovery(ref_chroma, live_base, speed, noise, label, jump=None, total_measures=None):
    """Führt Simulation mit RecoveryODTW durch.

    jump: Tuple (from_measure, to_measure) für Rückwärtssprung, oder None.
    """
    print(f"\n--- {label} [RECOVERY]: Tempo {speed*100:.0f}%, Noise {noise} ---")

    live_input = prepare_simulation_data(live_base, speed, noise)
    if jump:
        live_input = inject_jump_back(live_input, jump[0], jump[1], total_measures)
    dtw = RecoveryODTW(ref_chroma)

    positions = []
    costs = []
    recovery_points = []
    n_frames = live_input.shape[1]

    for i in range(n_frames):
        live_vec = live_input[:, i]
        pos, cost, recovered = dtw.step(live_vec)
        positions.append(pos)
        costs.append(cost)

        if recovered:
            recovery_points.append(i)

        if i % 300 == 0 or i == n_frames - 1:
            progress = (i / n_frames) * 100
            print(f"\rVerarbeite {n_frames} Frames... [{progress:5.1f}%]", end="")

    print(f"\n  Recoveries: {len(recovery_points)}")
    return positions, costs, recovery_points, n_frames

# --- PLOTTING FUNCTIONS ---

def plot_analysis(ref_chroma, live_base, use_recovery=False, jump=None):
    """
    Plottet detaillierte Debug-Analyse mit zwei Subplots:
    1. Tracking Position
    2. Globale Kosten (Moving Average)

    use_recovery: True = RecoveryODTW, False = DebugODTW
    jump: Tuple (from_%, to_%) für Rückwärtssprung, oder None
    """
    fig, (ax_path, ax_avg) = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)

    ref_len = ref_chroma.shape[1]
    total_measures = frames_to_measures(ref_len)
    WINDOW_SIZE = 300
    mode_name = "RecoveryODTW" if use_recovery else "DebugODTW"
    if jump:
        mode_name += f" + Sprung (Takt {jump[0]}->Takt {jump[1]})"

    for speed, noise, color, label in TEST_SCENARIOS:
        if use_recovery:
            pos, costs, recovery_points, input_len = run_simulation_recovery(
                ref_chroma, live_base, speed, noise, label, jump=jump,
                total_measures=total_measures
            )
            # glob für Moving Average = costs
            glob = costs
        else:
            pos, glob, loc, input_len = run_simulation_debug(
                ref_chroma, live_base, speed, noise, label
            )
            recovery_points = []

        # X-Achse: Alle Szenarien auf Referenz-Takte normalisiert
        # Dadurch liegen alle Geschwindigkeiten auf der gleichen Skala
        x_axis = np.linspace(0, total_measures, input_len)

        # Y-Achse: Positionen in Takte umrechnen
        pos_measures = [frames_to_measures(p) for p in pos]

        # Plot 1: Tracking Pfad
        ax_path.plot(x_axis, pos_measures, color=color, label=label, linewidth=1.5)

        # Recovery-Punkte markieren
        if recovery_points:
            rec_x = [x_axis[i] for i in recovery_points if i < len(x_axis)]
            rec_y = [pos_measures[i] for i in recovery_points if i < len(pos_measures)]
            ax_path.scatter(rec_x, rec_y, color=color, marker='X', s=100,
                           edgecolors='black', zorder=5)

        # Plot 2: Moving Average
        if len(glob) >= WINDOW_SIZE:
            glob_smooth = np.convolve(glob, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
            x_axis_smooth = np.linspace(0, total_measures, len(glob_smooth))
            ax_avg.plot(x_axis_smooth, glob_smooth, color=color, label=label, linewidth=2)

    # Formatierung
    ax_path.set_title("1. Tracking Position")
    ax_path.set_ylabel("Takt in Partitur")
    ax_path.set_xlabel("Takt im Input")
    ax_path.grid(True)

    # Ideallinie
    if jump:
        ax_path.plot(
            [0, jump[0], jump[0], total_measures],
            [0, jump[0], jump[1], total_measures],
            'k--', alpha=0.4, linewidth=2, label='Ideal (mit Sprung)'
        )
    else:
        ax_path.plot([0, total_measures], [0, total_measures], 'k--', alpha=0.3, label='Ideal')

    ax_path.legend(loc='upper left')

    ax_avg.set_title(f"2. Globale Kosten (Moving Average, {WINDOW_SIZE} Frames)")
    ax_avg.set_ylabel("Kosten")
    ax_avg.set_xlabel("Takt im Input")
    ax_avg.grid(True)
    ax_avg.legend(loc='upper left')

    plt.suptitle(f"ODTW Debug-Analyse ({mode_name})", fontsize=16)
    plt.show()

# --- MAIN ---

def main():
    """Hauptprogramm - führt Debug-Analyse durch"""
    parser = argparse.ArgumentParser(description="ODTW Robustness Testing")
    parser.add_argument("--recovery", action="store_true",
                        help="RecoveryODTW statt DebugODTW verwenden")
    parser.add_argument("--jump", nargs=2, type=int, metavar=("FROM", "TO"),
                        help="Rückwärtssprung simulieren: --jump 50 30 (von Takt 50 zu Takt 30)")
    parser.add_argument("--score", default=SCORE_FILE, help="Partitur (.npz)")
    parser.add_argument("--wav", default=LIVE_WAV_FILE, help="Live-Audio WAV")
    args = parser.parse_args()

    # Referenz-Chroma laden
    try:
        ref_chroma = load_score_chroma(args.score)
    except Exception as e:
        print(f"Fehler beim Laden der Referenz: {e}")
        return

    # WAV-Datei laden
    mode = "RecoveryODTW" if args.recovery else "DebugODTW"
    print(f"\n=== ODTW Robustness Testing ({mode}) ===")
    print("Lade Live-Audio von WAV...")
    try:
        live_base = load_chroma_from_wav(args.wav, save_npy=True)
    except FileNotFoundError:
        print(f"Fehler: {args.wav} nicht gefunden!")
        return
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # Jump-Parameter
    jump = tuple(args.jump) if args.jump else None
    if jump:
        print(f"Sprung-Simulation: Takt {jump[0]} -> Takt {jump[1]}")

    # Führe Debug-Analyse durch
    print(f"\nStarte Analyse mit {mode}...")
    plot_analysis(ref_chroma, live_base, use_recovery=args.recovery, jump=jump)

if __name__ == "__main__":
    main()
