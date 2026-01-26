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
SEARCH_WINDOW = 300    # Suchradius: Wir suchen nur +/- 80 Frames um die letzte Position
HOP_LENGTH = 512     # Wie viel wir im Fenster weiter springen (Overlap)
DAMPING_FACTOR = 0.99  # Dämpfungsfaktor für alte Kosten
WAIT_PENALTY = 0.6    # Strafe fürs Stehenbleiben
SKIP_PENALTY = 0.9      # Strafe für zu weite - erstmal hohe Strafe
BPM = 40
BEATS_PER_MEASURE = 4
SMOOTHING_WINDOW = 15 # Anzahl Frames für Moving Average (1 = aus)
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

# --- NEUE HILFSKLASSE FÜR RINGBUFFER ---
class AudioRingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
    
    def append(self, new_data):
        # Verschiebe Daten nach links: Das Alte fällt raus
        # new_data Länge muss HOP_LENGTH sein
        n = len(new_data)
        self.buffer = np.roll(self.buffer, -n)
        self.buffer[-n:] = new_data # Neue Daten ans Ende schreiben
        
    def get(self):
        return self.buffer

# --- MAIN LOOP (Korrigiert) ---

def main():
    print("\n--- Audio-Geräte ---")
    print(sd.query_devices())
    print("--------------------\n")

    try:
        ref_chroma = load_h_file_chroma("ScoreData.h")
        n_frames = ref_chroma.shape[1]
    except Exception as e:
        print(f"Fehler: {e}")
        return

    dtw_engine = StandardODTW(ref_chroma)
    
    # RINGBUFFER INITIALISIEREN
    # Der Buffer muss so groß sein wie die FFT braucht (BLOCK_SIZE = 4096)
    ring_buffer = AudioRingBuffer(BLOCK_SIZE)
    
    print(f"Starte Tracking bei {BPM} BPM...")
    print(f"INTERNER TAKT: Alle {HOP_LENGTH/SAMPLE_RATE*1000:.1f} ms ein Update.")

    try:
        # WICHTIG: Wir lesen jetzt nur HOP_LENGTH (512) Samples pro Schritt!
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=HOP_LENGTH) as stream:
            while True:
                # 1. Nur kleine Menge lesen (512 Samples -> 11ms)
                new_chunk, overflow = stream.read(HOP_LENGTH)
                new_data = new_chunk[:, 0] # Mono

                # 2. In den Ringbuffer schieben (damit wir 4096 für die FFT haben)
                ring_buffer.append(new_data)
                
                # 3. FFT auf dem vollen 4096 Buffer machen
                audio_for_fft = ring_buffer.get()

                # Energie Check
                rms = np.sqrt(np.mean(audio_for_fft**2))
                energy_marker = " "
                if rms > 0.1: energy_marker = "!!!" 
                elif rms > 0.01: energy_marker = "*" 
                elif rms < 0.001: energy_marker = "_" 
                
                # 4. Chroma berechnen
                # WICHTIG: n_fft=BLOCK_SIZE. Wir übergeben genau BLOCK_SIZE Daten.
                # hop_length ist hier egal, da wir nur 1 Frame berechnen wollen.
                chroma_stft = librosa.feature.chroma_stft(
                    y=audio_for_fft, 
                    sr=SAMPLE_RATE, 
                    n_fft=BLOCK_SIZE, 
                    hop_length=BLOCK_SIZE+1, # Trick: Nur 1 Frame berechnen
                    center=False, 
                    tuning=0
                )
                
                # Das Ergebnis ist (12, 1) -> Flatten zu (12,)
                live_vec = np.mean(chroma_stft, axis=1)
                
                # 5. ODTW Schritt (jetzt läuft er mit ca. 86 Hz!)
                pos_index, current_cost = dtw_engine.step(live_vec)
                
                measure, beat = calculate_current_measure(pos_index, SAMPLE_RATE, HOP_LENGTH, BPM, BEATS_PER_MEASURE)
                
                # Visualisierung (etwas bremsen, sonst flimmert die Konsole)
                # Wir geben nur jeden 10. Frame aus
                if pos_index % 2 == 0: 
                    progress_percent = (pos_index / n_frames) * 100
                    bar_len = 30
                    filled = int(bar_len * pos_index // n_frames)
                    bar = '█' * filled + '-' * (bar_len - filled)
                    print(f"\rTakt {measure:03d}.{beat} | Cost: {current_cost:6.2f} | {bar} {progress_percent:.1f}%", end="")

    except KeyboardInterrupt:
        print("\nBeendet.")
    except Exception as e:
        print(f"\nFehler: {e}")

if __name__ == "__main__":
    main()