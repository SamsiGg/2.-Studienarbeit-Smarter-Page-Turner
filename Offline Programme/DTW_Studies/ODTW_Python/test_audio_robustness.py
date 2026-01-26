import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from dtw_live_test import StandardODTW, load_h_file_chroma

# --- KONFIGURATION ---
# Die Datei mit den Referenz-Vektoren (aus MusicXML)
SCORE_FILE = "ScoreData.h"

# Die Datei mit den Live-Vektoren (aus der Audio-Aufnahme, als .npy gespeichert)
LIVE_NPY_FILE = "Fiocco-Live (40bpm)_chroma.npy" 

# Musikalische Parameter (Müssen zur Analyse passen!)
SAMPLE_RATE = 44100
HOP_LENGTH = 512
BPM = 40  # Tempo des Stücks in der Partitur
BEATS_PER_MEASURE = 4

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
    # Wir zoomen nur entlang der Zeitachse (Achse 1), nicht entlang der Tonhöhe (Achse 0)
    # Wenn speed_factor = 2.0 (doppelt so schnell), muss die Länge 0.5x sein.
    zoom_factor = 1 / speed_factor
    
    # order=1 ist lineare Interpolation (ausreichend für Chroma)
    return zoom(chroma, (1, zoom_factor), order=1)

def run_simulation(live_chroma_base, ref_chroma, speed_factor=1.0, noise_level=0.0):
    print(f"\n--- Simulation: Tempo {speed_factor*100:.0f}%, Noise {noise_level} ---")
    
    # 1. Time Stretching auf die Vektoren anwenden
    if speed_factor != 1.0:
        # Wir übergeben eine Kopie, damit das Original nicht verändert wird
        live_chroma = stretch_chroma(live_chroma_base, speed_factor)
    else:
        live_chroma = live_chroma_base.copy()
        
    # 2. Rauschen hinzufügen (Robustheitstest)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, live_chroma.shape)
        live_chroma = live_chroma + noise
        # Negative Werte verhindern und Clipping
        live_chroma = np.clip(live_chroma, 0, 1)
        
        # Optional: Neu normalisieren, damit ODTW saubere Vektoren bekommt
        norms = np.linalg.norm(live_chroma, axis=0) + 1e-9
        live_chroma = live_chroma / norms

    # 3. ODTW initialisieren
    dtw = StandardODTW(ref_chroma)
    
    positions = []
    n_live_frames = live_chroma.shape[1] # Zeitachse ist Dimension 1
    
    # 4. Schrittweise simulieren
    print(f"Verarbeite {n_live_frames} Frames...")
    
    # Transponieren für die Iteration (Zeit soll erste Dimension sein)
    live_chroma_steps = live_chroma.T 
    
    for live_vec in live_chroma_steps:
        # Achtung: step() erwartet einen einzelnen Vektor (12,)
        pos, cost = dtw.step(live_vec) 
        positions.append(pos)
        
    return positions, n_live_frames

def plot_results():
    # 1. Daten laden
    try:
        ref_chroma = load_h_file_chroma(SCORE_FILE)
        live_chroma_base = np.load(LIVE_NPY_FILE)
        
        # Falls Shape (N, 12) ist, transponieren zu (12, N) für interne Verarbeitung
        if live_chroma_base.shape[0] > 12 and live_chroma_base.shape[1] == 12:
            live_chroma_base = live_chroma_base.T
            
    except FileNotFoundError as e:
        print(f"Fehler: {e}")
        return

    # Referenzlänge für Plot-Achsen
    ref_frames = ref_chroma.shape[1]
    total_measures_score = frames_to_measures(ref_frames)
    print(f"Länge der Partitur: {ref_frames} Frames = {total_measures_score:.2f} Takte")

    plt.figure(figsize=(10, 8))
    
    # --- PLOTTING HELPER ---
    def plot_line(speed, noise, label, color, style='-'):
        path_frames, len_input = run_simulation(live_chroma_base, ref_chroma, speed, noise)
        
        # Um die Linien vergleichbar zu machen, normieren wir die X-Achse.
        # 0 = Start der Aufnahme, Ende = Ende der Aufnahme.
        # Da wir wissen, dass die Aufnahme (unabhängig vom Tempo) das GANZE Stück beinhaltet,
        # mappen wir das Ende der Aufnahme auf das Ende des Stücks (in Takten).
        
        # X-Achse: Prozentualer Fortschritt der Aufnahme skaliert auf Takte
        x_axis = np.linspace(0, total_measures_score, len_input)
        
        # Y-Achse: Vom ODTW erkannte Position (in Frames) umgerechnet in Takte
        y_axis = [frames_to_measures(p) for p in path_frames]
        
        plt.plot(x_axis, y_axis, label=label, color=color, linestyle=style, linewidth=2)

    # Szenarien testen
    
    # 1. Referenz (Ideal: 1.0x Speed, kein Noise)
    plot_line(speed=1.0, noise=0.0, label="Normal (1.0x)", color='green')
    
    # 2. Schnell spielen (1.2x) - Prüft, ob ODTW hinterherkommt
    plot_line(speed=1.2, noise=0.1, label="Schnell (1.2x) + Noise", color='red', style='--')
    
    # 3. Langsam spielen (0.8x) - Prüft, ob ODTW nicht zu schnell vorauseilt
    plot_line(speed=0.8, noise=0.1, label="Langsam (0.8x) + Noise", color='blue', style=':')

    # Ideale Diagonale
    plt.plot([0, total_measures_score], [0, total_measures_score], 'k-', alpha=0.3, label="Ideal")
    
    plt.title(f"ODTW Robustheitstest (.npy Input)\nBPM: {BPM}, Noise-Level simuliert schlechtes Audio")
    plt.xlabel("Fortschritt im Input (normiert auf Taktlänge)")
    plt.ylabel("Erkannte Position (Takt in Partitur)")
    
    plt.axis('equal') 
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()