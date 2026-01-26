import os
import argparse
import librosa
import numpy as np

# --- KONFIGURATION ---
# Diese Werte müssen EXAKT mit deinem MusicXML-Skript übereinstimmen,
# damit die Vektoren vergleichbar sind.
SAMPLE_RATE = 44100
FFT_SIZE = 4096
HOP_LENGTH = 512

def wav_to_chroma(input_file, output_format="npy"):
    """
    Liest eine .wav Datei, berechnet die Chroma-Vektoren (L2-normalisiert)
    und speichert sie ab.
    """
    if not os.path.exists(input_file):
        print(f"❌ Fehler: Datei '{input_file}' nicht gefunden.")
        return

    base_name = os.path.splitext(input_file)[0]
    
    print(f"📂 Lade Audio: {input_file}")
    
    # 1. Audio laden (Mono, 44.1kHz)
    try:
        y, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"❌ Fehler beim Laden der Audio: {e}")
        return

    print("📊 Berechne Chroma Features...")
    
    

    # 2. Chroma Berechnung (Identisch zum MusicXML Skript)
    # n_chroma=12 entspricht den 12 Halbtönen (C, C#, D...)
    chroma_raw = librosa.feature.chroma_stft(
        y=y, 
        sr=sr, 
        n_fft=FFT_SIZE, 
        hop_length=HOP_LENGTH, 
        n_chroma=12
    )
    
    # 3. Transponieren und Normalisieren (L2 Norm)
    # Form ändern von [12, Zeit] zu [Zeit, 12]
    chroma_steps = chroma_raw.T 
    
    # L2-Normalisierung (Vektorlänge = 1)
    # Das macht die Erkennung robuster gegen Lautstärkeunterschiede
    norms = np.linalg.norm(chroma_steps, axis=1, keepdims=True)
    chroma_l2 = chroma_steps / (norms + 1e-9) # +1e-9 verhindert Division durch Null

    num_frames = chroma_l2.shape[0]
    print(f"✅ Berechnung fertig! Anzahl Frames: {num_frames}")

    # 4. Speichern
    if output_format == "npy":
        output_file = base_name + "_chroma.npy"
        np.save(output_file, chroma_l2)
        print(f"💾 Gespeichert als NumPy Binary: {output_file}")
    
    elif output_format == "csv":
        output_file = base_name + "_chroma.csv"
        # Speichern als lesbare Textdatei (Komma-getrennt)
        np.savetxt(output_file, chroma_l2, delimiter=",", fmt="%.4f")
        print(f"💾 Gespeichert als CSV: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erstellt Chroma-Vektoren direkt aus einer WAV-Datei.")
    parser.add_argument("input_file", help="Pfad zur .wav Datei")
    parser.add_argument("--format", type=str, choices=["npy", "csv"], default="npy", help="Speicherformat: 'npy' (Python) oder 'csv' (Text/Excel)")
    
    args = parser.parse_args()
    
    wav_to_chroma(args.input_file, args.format)