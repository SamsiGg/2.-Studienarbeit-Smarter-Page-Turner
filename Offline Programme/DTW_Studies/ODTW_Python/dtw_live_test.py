import numpy as np
import librosa
import sounddevice as sd
import scipy.spatial.distance as dist
import time
import re
from collections import deque

# --- KONFIGURATION ---
SAMPLE_RATE = 44100  # Weniger reicht oft für Chroma und spart Rechenleistung
BLOCK_SIZE = 4096     # Größe des Audio-Schnipsels (Latzenz vs. Genauigkeit)
SEARCH_WINDOW = 80    # Suchradius: Wir suchen nur +/- 80 Frames um die letzte Position
HOP_LENGTH = 512     # Wie viel wir im Fenster weiter springen (Overlap)
DAMPING_FACTOR = 0.96  # Dämpfungsfaktor für alte Kosten
WAIT_PENALTY = 0.3    # Strafe fürs Stehenbleiben
SKIP_PENALTY = 0.2      # Strafe für zu weite - erstmal hohe Strafe
BPM = 40
BEATS_PER_MEASURE = 4
SMOOTHING_WINDOW = 5 # Anzahl Frames für Moving Average (1 = aus)
# ----------------------

def load_h_file_chroma(filename):
    print(f"Lade {filename}...")
    with open(filename, 'r') as f:
        content = f.read()
    
    # 1. Den Start der Chroma-Daten finden
    # Wir suchen nach dem Variablennamen "score_chroma"
    keyword = "score_chroma"
    start_pos = content.find(keyword)
    
    if start_pos == -1:
        raise ValueError(f"Konnte '{keyword}' in der Datei nicht finden!")
        
    # Wir schneiden alles vor der Variable ab (damit ignorieren wir num_pages, score_len etc.)
    content_chroma = content[start_pos:]
    
    # Jetzt suchen wir die erste geschweifte Klammer { NACH dem Variablennamen
    array_start = content_chroma.find('{')
    array_end = content_chroma.rfind('}') # Die allerletzte Klammer
    
    # Nur den Inhalt zwischen den Klammern nehmen
    data_string = content_chroma[array_start:array_end+1]
    
    # 2. Bereinigen
    clean_content = data_string.replace('f', '').replace('{', '').replace('}', '').replace(';', '')
    tokens = clean_content.replace(',', ' ').split()
    
    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            continue
            
    # 3. Validierung
    if len(values) == 0:
        raise ValueError("Keine Zahlen im Chroma-Bereich gefunden!")
        
    if len(values) % 12 != 0:
        print(f"WARNUNG: Anzahl Werte ({len(values)}) nicht durch 12 teilbar. Schneide ab.")
        cut_len = (len(values) // 12) * 12
        values = values[:cut_len]
        
    # Shape (N, 12) -> Transponieren zu (12, N)
    arr = np.array(values).reshape(-1, 12).T
    
    print("--- DATEN CHECK ---")
    print(f"Erster Vektor (Sample): {arr[:, 0]}")
    print(f"L2-Norm Frame 0: {np.linalg.norm(arr[:, 0]):.4f}")
    
    return arr

# --- Aktualisierte DTW Logik mit deinen Ideen ---

class StandardODTW:
    def __init__(self, reference_chroma):
        # Referenz (Muss L2 normalisiert sein!)
        self.ref = reference_chroma 
        self.n_frames_ref = self.ref.shape[1]
        
        self.current_position_index = 0
        
        # Akkumulierte Kosten initialisieren
        self.accumulated_costs = np.full(self.n_frames_ref, np.inf)
        self.accumulated_costs[0] = 0 
        
        # --- EINSTELLUNGEN (Additive Penalties) ---
        # Standard Schritt (Diagonal i-1 -> i) kostet nichts extra
        self.penalty_diagonal = 0.0 
        
        # Warten (Vertikal i -> i) wird bestraft, um Vorwärtsbewegung zu fördern
        # Ein Wert zwischen 0.1 und 0.5 der max. Distanz (2.0) ist üblich
        self.penalty_wait = WAIT_PENALTY
        
        # Überspringen (i-2 -> i) wird stärker bestraft
        self.penalty_skip = SKIP_PENALTY

        self.damping = DAMPING_FACTOR

        self.chroma_buffer = deque(maxlen=SMOOTHING_WINDOW)

    def step(self, live_chroma_raw):
        """
        live_chroma_vector: Rohdaten, muss noch normalisiert werden
        """
        
        # 1. Glättung (Moving Average)
        self.chroma_buffer.append(live_chroma_raw)
        
        # Durchschnitt berechnen
        # axis=0 bedeutet, wir mitteln über die Zeitachse des Buffers
        avg_chroma = np.mean(np.array(self.chroma_buffer), axis=0)
        
        # 2. Normalisierung (JETZT erst normalisieren!)
        norm = np.linalg.norm(avg_chroma)
        if norm > 0.0001:
            live_vec_norm = avg_chroma / norm
        else:
            live_vec_norm = avg_chroma
            
        # Suchfenster definieren
        start_scan = max(0, self.current_position_index - SEARCH_WINDOW)
        end_scan = min(self.n_frames_ref, self.current_position_index + SEARCH_WINDOW)
        
        new_costs = np.full(self.n_frames_ref, np.inf)
        best_local_cost = np.inf
        best_index = self.current_position_index

        for i in range(start_scan, end_scan):
            # A: Lokale Distanz (Cosine Distance: 0.0 bis 2.0)
            # 1 - DotProduct funktioniert nur bei normalisierten Vektoren!
            dot_prod = np.dot(live_vec_norm, self.ref[:, i])
            # Clampen für numerische Stabilität
            dot_prod = max(-1.0, min(1.0, dot_prod))
            local_dist = 1.0 - dot_prod
            
            # B: Pfad-Auswahl (Minimum der Vorgänger + Additive Penalty)
            
            # 1. Warten (von sich selbst / i kommen)
            cost_wait = self.accumulated_costs[i] + self.penalty_wait
            
            # 2. Schritt (Diagonal / von i-1 kommen)
            cost_step = np.inf
            if i > 0:
                cost_step = self.accumulated_costs[i-1] + self.penalty_diagonal
            
            # 3. Skip (von i-2 kommen) - Optionales Feature
            cost_skip = np.inf
            if i > 1:
                cost_skip = self.accumulated_costs[i-2] + self.penalty_skip
            
            # Finde den günstigsten Pfad
            min_prev_cost = min(cost_wait, cost_step, cost_skip)
            
            # C: Gesamtkosten
            # Formel: D(i,j) = dist(i,j) + min(Vorgänger)
            new_costs[i] = local_dist + (min_prev_cost * self.damping)

            # D: Besten Index für diesen Zeitschritt finden
            if new_costs[i] < best_local_cost:
                best_local_cost = new_costs[i]
                best_index = i
        
        # Zustand updaten
        self.accumulated_costs = new_costs
        self.current_position_index = best_index
        
        return best_index, best_local_cost
    
def get_live_chroma(audio_buffer, sr):
    """Extrahiert Chroma aus dem Audio-Buffer (Wrapper für Librosa)"""
    # Auf dem Teensy wird das durch die Audio-Library (FFT) ersetzt
    chroma = librosa.feature.chroma_stft(y=audio_buffer, sr=sr, n_fft=len(audio_buffer), hop_length=len(audio_buffer)+1, center=False)
    # Librosa gibt (12, 1) zurück, wir wollen (12,)
    return np.mean(chroma, axis=1)

def generate_mock_reference():
    """Erstellt eine Fake-Partitur (Sinus-Töne, die die Tonleiter hochgehen)"""
    print("Generiere Mockup-Daten (C-Dur Tonleiter)...")
    dummy_audio = np.array([])
    # Erzeuge Audio: C, D, E, F, G...
    freqs = [261.6, 293.7, 329.6, 349.2, 392.0, 440.0] 
    for f in freqs:
        t = np.linspace(0, 1.0, int(SAMPLE_RATE * 1.0)) # 1 Sekunde pro Ton
        tone = 0.5 * np.sin(2 * np.pi * f * t)
        dummy_audio = np.concatenate((dummy_audio, tone))
    
    # Referenz-Chroma berechnen
    ref_chroma = librosa.feature.chroma_stft(y=dummy_audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512)
    return ref_chroma

def calculate_current_measure(frame_index, sr, hop_length, bpm, beats_per_bar):
    """Rechnet Frame-Index in Taktnummer um"""
    # 1. Zeit in Sekunden
    time_sec = (frame_index * hop_length) / sr
    
    # 2. Zeit pro Takt (in Sekunden)
    # Bei 40 BPM sind das 1.5 Sekunden pro Schlag -> 6 Sekunden pro 4/4 Takt
    sec_per_beat = 60.0 / bpm
    sec_per_measure = sec_per_beat * beats_per_bar
    
    # 3. Aktueller Takt (Start bei 1)
    measure = int(time_sec / sec_per_measure) + 1
    
    # Optional: Schlag im Takt (1, 2, 3, 4)
    beat_in_measure = int((time_sec % sec_per_measure) / sec_per_beat) + 1
    
    return measure, beat_in_measure

# --- MAIN LOOP ---

def main():
    # --- 1. Audio-Geräte Check ---
    print("\n--- Verfügbare Audio-Geräte ---")
    print(sd.query_devices())
    # Falls das falsche Mikro gewählt wird, ändere 'device=X' im InputStream unten!
    print("-------------------------------\n")

    # --- 2. Referenz laden ---
    try:
        ref_chroma = load_h_file_chroma("ScoreData.h")
        n_frames = ref_chroma.shape[1]
        print(f"Referenz geladen: {n_frames} Frames.")
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # ODTW Initialisieren
    dtw_engine = StandardODTW(ref_chroma)
    
    # Parameter Setup für saubere Anzeige
    print(f"Starte Tracking bei {BPM} BPM...")
    print("Format: [Takt . Schlag] (Energie-Level) | Fortschrittsbalken")

    # --- 3. Live Loop ---
    try:
        # device=1  <-- HIER KANNST DU DEN INDEX ÄNDERN, falls das falsche Mikro an ist
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as stream:
            while True:
                # A: Audio lesen
                audio_chunk, overflow = stream.read(BLOCK_SIZE)
                audio_data = audio_chunk[:, 0] # Mono

                # B: Energie berechnen (RMS) für Debugging
                # Einfache Formel: Wurzel aus dem Durchschnitt der Quadrate
                rms = np.sqrt(np.mean(audio_data**2))
                
                # C: Wenn Energie sehr niedrig ist -> Warnung (Stille)
                energy_marker = " "
                if rms > 0.1: energy_marker = "!!!" # Laut
                elif rms > 0.01: energy_marker = "*" # Signal da
                elif rms < 0.001: energy_marker = "_" # Stille
                
                # D: Chroma berechnen & ODTW Schritt
                # Wir müssen HOP_LENGTH an librosa übergeben, damit es konsistent ist
                # D: Chroma berechnen & ODTW Schritt
                # tuning=0 verhindert die Warnung bei Stille
                chroma_stft = librosa.feature.chroma_stft(
                    y=audio_data, 
                    sr=SAMPLE_RATE, 
                    n_fft=BLOCK_SIZE, 
                    hop_length=HOP_LENGTH+1, 
                    center=False, 
                    tuning=0
                )# Mittelwert über das Fenster (falls librosa mehr als 1 Frame zurückgibt)
                live_vec = np.mean(chroma_stft, axis=1)
                
                pos_index, current_cost = dtw_engine.step(live_vec)
                
                # E: Takt Berechnung
                measure, beat = calculate_current_measure(pos_index, SAMPLE_RATE, HOP_LENGTH, BPM, BEATS_PER_MEASURE)
                
                # F: Visualisierung
                # Um die Konsole nicht zu fluten, nutzen wir \r (Carriage Return)
                progress_percent = (pos_index / n_frames) * 100
                bar_len = 30
                filled = int(bar_len * pos_index // n_frames)
                bar = '█' * filled + '-' * (bar_len - filled)
                
                # Anzeige:
                # Takt.Schlag | Energie-Wert (Balken) | Fortschritt
                print(f"\rTakt {measure:03d}.{beat} | Cost: {current_cost:6.2f} | Mic: {rms:.4f} [{energy_marker}] | {bar} {progress_percent:.1f}%", end="")

    except KeyboardInterrupt:
        print("\nBeendet.")
    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBeendet.")