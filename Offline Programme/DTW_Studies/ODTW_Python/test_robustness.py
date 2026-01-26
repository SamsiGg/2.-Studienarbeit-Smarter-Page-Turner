import numpy as np
import librosa
import matplotlib.pyplot as plt
from dtw_live_test import StandardODTW, load_h_file_chroma

# Konfiguration
AUDIO_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/MusescoreToChroma/Fiocco.wav"
SCORE_FILE = "ScoreData.h"

# --- MUSIKALISCHE PARAMETER (Müssen stimmen!) ---
SAMPLE_RATE = 44100
HOP_LENGTH = 2048
BPM = 40
BEATS_PER_MEASURE = 4

def frames_to_measures(n_frames):
    """Rechnet eine Anzahl an Frames in Takte um"""
    # Dauer eines Frames in Sekunden
    sec_per_frame = HOP_LENGTH / SAMPLE_RATE
    # Dauer der gesamten Frames
    total_seconds = n_frames * sec_per_frame
    
    # Dauer eines Taktes in Sekunden (60/BPM * Schläge)
    sec_per_measure = (60.0 / BPM) * BEATS_PER_MEASURE
    
    return total_seconds / sec_per_measure

def run_simulation(speed_factor=1.0, noise_level=0.2):
    print(f"\n--- Simulation: Tempo {speed_factor*100:.0f}%, Noise {noise_level} ---")
    
    # Audio laden (ohne Duration-Limit, ganze Datei)
    y, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)
    
    if speed_factor != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed_factor)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(y))
        y = y + noise

    chroma_full = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=HOP_LENGTH, tuning=0)
    
    ref_chroma = load_h_file_chroma(SCORE_FILE)
    dtw = StandardODTW(ref_chroma)
    
    positions = []
    n_live_frames = chroma_full.shape[1]
    
    print(f"Verarbeite {n_live_frames} Frames...")
    
    for i in range(n_live_frames):
        live_vec = chroma_full[:, i]
        pos, cost = dtw.step(live_vec) # step gibt (index, cost) zurück
        positions.append(pos)
        
    return positions, n_live_frames

def plot_results():
    # Referenzlänge bestimmen (für die Ideal-Linie und Y-Achse)
    ref_chroma = load_h_file_chroma(SCORE_FILE)
    ref_frames = ref_chroma.shape[1]
    
    # Gesamtzahl der Takte in der Partitur berechnen
    total_measures_score = frames_to_measures(ref_frames)
    print(f"Länge der Partitur: {ref_frames} Frames = {total_measures_score:.2f} Takte")

    plt.figure(figsize=(10, 8)) # Etwas quadratischer für bessere Diagonale
    
    # --- PLOTTING FUNKTION ---
    def plot_line(speed, label, color, style='-'):
        path_frames, len_input_frames = run_simulation(speed)
        
        # X-Achse: Die gespielte Zeit in Takte umrechnen
        # Da wir die ganze Datei spielen, entspricht das Ende genau der Länge des Stücks (interpretiert)
        # Bei 1.0x Speed ist Input-Länge = Score-Länge.
        # Bei 1.2x Speed sind wir "schneller fertig", aber der INHALT sind immer noch X Takte.
        # Daher normieren wir die X-Achse auf die Anzahl der Takte der PARTITUR.
        x_axis = np.linspace(0, total_measures_score, len_input_frames)
        
        # Y-Achse: Die erkannten Frame-Positionen in Takte umrechnen
        y_axis = [frames_to_measures(p) for p in path_frames]
        
        plt.plot(x_axis, y_axis, label=label, color=color, linestyle=style, linewidth=2)

    # 1. Normal
    plot_line(1.0, "Normal (1.0x)", 'green')
    
    # 2. Schnell
    plot_line(1.2, "Schnell (1.2x)", 'red', '--')
    
    # 3. Langsam
    plot_line(0.8, "Langsam (0.8x)", 'blue', ':')

    # Ideal-Linie (Diagonale von 0 bis Ende)
    plt.plot([0, total_measures_score], [0, total_measures_score], 'k-', alpha=0.3, label="Ideal")
    
    plt.title(f"DTW Robustheit (BPM: {BPM}, 4/4 Takt)")
    plt.xlabel("Gespielter Takt (Zeit)")
    plt.ylabel("Erkannte Position (Takt in Partitur)")
    
    # Achsen gleich skalieren, damit Diagonale wirklich 45 Grad ist
    plt.axis('equal') 
    plt.xlim(0, total_measures_score)
    plt.ylim(0, total_measures_score)
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on() # Hilfsraster für Takte
    plt.show()

if __name__ == "__main__":
    plot_results()