# =============================================================================
# audio_generator.py – Chroma-Vektoren → WAV-Melodie (dominanter Ton)
# =============================================================================
# Nutzung:
#   python audio_generator.py ../data/generated/Fiocco.npz
#   python audio_generator.py ../data/generated/Fiocco.npz --out melodie.wav
#
# Unterstützte Formate: .npz (Score Pipeline), .npy, .h (Legacy)
# =============================================================================

import argparse
import os
import re
import sys
import numpy as np
from scipy.io import wavfile

# --- KONFIGURATION ---
# Falls nicht per Argument überschrieben, wird dies als Output genutzt
DEFAULT_OUTPUT_FILENAME = 'dominant_tone_melody.wav'

# WICHTIG: Diese Rate muss zur Hop-Length der Analyse passen!
# Bei sr=44100 und hop=512 ist die Rate ca. 86.13 Hz.
# Wenn du hier 172 nutzt, spielt es doppelt so schnell (oder für hop=256).
VECTORS_PER_SECOND = 86.13  
SAMPLE_RATE = 44100

# Frequenzen (Mittlere Oktave C4 - H4)
NOTE_FREQS = np.array([
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 
    369.99, 392.00, 415.30, 440.00, 466.16, 493.88
])

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]

def load_chroma_vectors(filename):
    """
    Entscheidet anhand der Dateiendung, ob eine .npy oder eine Textdatei (.h) geladen wird.
    Gibt eine Matrix der Form (N, 12) zurück.
    """
    if not os.path.exists(filename):
        print(f"❌ FEHLER: Datei '{filename}' nicht gefunden.")
        sys.exit(1)

    # Dateiendung prüfen
    _, ext = os.path.splitext(filename)

    if ext.lower() == '.npz':
        print(f"📂 Lade NumPy-Archiv: {filename} ...")
        try:
            archive = np.load(filename)
            data = archive['chroma']  # Shape (12, N)
            print(f"  Chroma: {data.shape}, Seitengrenzen: {archive['page_end_indices']}")
            # Transponieren zu (N, 12) für dieses Skript
            return data.T
        except Exception as e:
            print(f"❌ Fehler beim Laden der .npz Datei: {e}")
            sys.exit(1)

    elif ext.lower() == '.npy':
        print(f"📂 Lade NumPy Binärdatei: {filename} ...")
        try:
            data = np.load(filename)
            # Sicherstellen, dass die Form (Zeit, 12) ist
            if data.ndim == 2:
                if data.shape[1] == 12:
                    return data
                elif data.shape[0] == 12:
                    print("⚠️  Format war (12, Zeit), wurde transponiert zu (Zeit, 12).")
                    return data.T

            print(f"❌ FEHLER: Unerwartetes Array-Format: {data.shape}. Erwartet (N, 12).")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Fehler beim Laden der .npy Datei: {e}")
            sys.exit(1)

    else:
        # Fallback: Die alte Text-Parsing Methode für ScoreData.h
        print(f"📄 Lade Text/Header Datei: {filename} ...")
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Fehler beim Lesen der Datei: {e}")
            sys.exit(1)

        raw_values = re.findall(r'([0-9]+\.[0-9]+)', content)
        if not raw_values:
            print("❌ FEHLER: Keine Zahlen in der Datei gefunden.")
            sys.exit(1)
            
        float_values = np.array([float(x) for x in raw_values], dtype=np.float32)
        
        # Auf glatte 12er Blöcke prüfen
        num_vectors = len(float_values) // 12
        if num_vectors == 0:
            print("❌ FEHLER: Zu wenige Daten für Vektoren.")
            sys.exit(1)

        return float_values[:num_vectors*12].reshape((num_vectors, 12))

def main():
    # Argumente parsen
    parser = argparse.ArgumentParser(description="Erstellt eine Melodie aus Chroma-Vektoren (.npz, .npy oder .h).")
    parser.add_argument("input_file", help="Pfad zur Eingabedatei (.npz, .npy oder .h)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_FILENAME, help="Name der Ausgabedatei (.wav)")
    
    args = parser.parse_args()

    # 1. Daten laden (Universal-Funktion)
    chroma_matrix = load_chroma_vectors(args.input_file)
    num_vectors = len(chroma_matrix)
    
    print(f"✅ Daten geladen. Verarbeite {num_vectors} Vektoren...")
    print("Modus: NUR DOMINANTER TON (1 aus 12)")

    # 2. Den Gewinner pro Vektor finden
    winner_indices = np.argmax(chroma_matrix, axis=1)
    
    # Die Lautstärke des Gewinners holen
    winner_volumes = chroma_matrix[np.arange(num_vectors), winner_indices]

    # Diagnose-Ausgabe
    print("\n--- Auszug der Melodie-Erkennung ---")
    prev_note = -1
    change_count = 0
    for i, idx in enumerate(winner_indices):
        if idx != prev_note:
            if change_count < 10: # Begrenzte Ausgabe
                print(f"Frame {i}: Wechsel zu {NOTE_NAMES[idx]} (Vol: {winner_volumes[i]:.2f})")
            prev_note = idx
            change_count += 1
    print(f"Insgesamt {change_count} Tonwechsel erkannt.\n")

    # 3. Audio Generierung
    samples_per_vector = SAMPLE_RATE / VECTORS_PER_SECOND
    
    # Frequenzen und Lautstärke auf Audio-Länge strecken
    target_freqs = np.repeat([NOTE_FREQS[i] for i in winner_indices], int(samples_per_vector))
    target_vols = np.repeat(winner_volumes, int(samples_per_vector))
    
    # 4. Synthese (Phasen-Akkumulation gegen Knacken)
    phase_increment = 2 * np.pi * target_freqs / SAMPLE_RATE
    phases = np.cumsum(phase_increment)
    
    audio = np.sin(phases) * target_vols

    # Normalisieren (Headroom lassen mit 0.9)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Speichern
    wavfile.write(args.out, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"🎉 Fertig! Datei gespeichert als: {args.out}")

if __name__ == "__main__":
    main()