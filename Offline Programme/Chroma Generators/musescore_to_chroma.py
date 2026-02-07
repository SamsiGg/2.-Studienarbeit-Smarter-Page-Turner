import os
import sys
import argparse
import subprocess
import music21
import librosa
import numpy as np

# --- KONFIGURATION ---
SOUNDFONT_PATH = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/MuseScore_General.sf2" 
SAMPLE_RATE = 44100
FFT_SIZE = 4096
HOP_LENGTH = 512

INSTRUMENTS = {
    "piano": 0, "violin": 40, "viola": 41, "cello": 42, 
    "contrabass": 43, "flute": 73, "clarinet": 71, "trumpet": 56, "guitar": 24
}

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

def set_tempo_hard(score, target_bpm, midi_program):
    """
    Löscht alle alten Tempos und erzwingt das neue.
    Verändert KEINE Noten/Akkorde.
    """
    print(f"🧹 Setze Tempo hart auf {target_bpm} BPM...")
    
    part = score.parts[0]
    
    # 1. ALLE alten Tempo-Informationen löschen (rekursiv)
    # Das entfernt alle versteckten 120 BPM Marker aus MuseScore
    try:
        for el in part.recurse():
            if 'MetronomeMark' in el.classes:
                el.activeSite.remove(el)
    except:
        pass # Falls nichts da ist, auch gut

    # 2. Neues Tempo setzen (Ganz am Anfang des Parts)
    new_tempo = music21.tempo.MetronomeMark(number=target_bpm)
    part.insert(0, new_tempo)
    
    # Zur Sicherheit auch im ersten Takt explizit setzen (Doppelt hält besser für MIDI Export)
    first_measure = part.getElementsByClass('Measure').first()
    if first_measure:
        first_measure.insert(0, new_tempo)

    # 3. Instrument setzen
    for el in part.recurse():
        if 'Instrument' in el.classes:
            el.activeSite.remove(el)
            
    new_inst = music21.instrument.Instrument()
    new_inst.midiProgram = midi_program
    part.insert(0, new_inst)

    return score

def get_interactive_page_turns(score, sr, hop_length, bpm):
    print("\n📖 --- SEITEN-LAYOUT KONFIGURATION ---")
    page_end_indices = []
    part = score.parts[0]
    frames_per_beat = (60.0 / bpm) * (sr / hop_length)

    page_counter = 1
    while True:
        try:
            user_input = input(f"Welches ist die LETZTE Taktnummer auf Seite {page_counter}? (oder 'x'): ")
            if user_input.lower() == 'x': break
            
            measure_num = int(user_input)
            found = False
            
            # Suche Takt rekursiv
            for m in part.recurse():
                if 'Measure' in m.classes and m.number == measure_num:
                    abs_start = m.getOffsetInHierarchy(score)
                    end_beat = abs_start + m.duration.quarterLength
                    
                    frame_index = int(end_beat * frames_per_beat)
                    page_end_indices.append(frame_index)
                    print(f"   -> Seite {page_counter} Ende: Takt {measure_num} (Frame {frame_index})")
                    found = True
                    page_counter += 1
                    break
            
            if not found: print(f"⚠️  Takt {measure_num} nicht gefunden!")
        except ValueError: print("Zahl eingeben.")

    return page_end_indices

def musicxml_to_teensy(input_file, bpm, instrument_name):
    base_name = os.path.splitext(input_file)[0]
    midi_debug_file = base_name + "_CHECK_THIS.mid" 
    wav_file = base_name + ".wav"
    header_file = "ScoreData.h"

    midi_program = INSTRUMENTS.get(instrument_name.lower(), 40)
    print(f"🚀 Starte: {input_file} @ {bpm} BPM")

    try:
        score = music21.converter.parse(input_file)
        
        # NUR TEMPO FIXEN (Noten bleiben original)
        score = set_tempo_hard(score, bpm, midi_program)

        page_indices = get_interactive_page_turns(score, SAMPLE_RATE, HOP_LENGTH, bpm)

        print(f"\n🎹 Speichere MIDI: {midi_debug_file}")
        score.write('midi', fp=midi_debug_file)

        print("🎻 Synthetisiere Audio...")
        if not os.path.exists(SOUNDFONT_PATH):
            print(f"🛑 FEHLER: Soundfont fehlt!")
            return

        fs_cmd = [
            "fluidsynth", "-ni", "-g", "1.0", "-F", wav_file, "-r", str(SAMPLE_RATE),
            SOUNDFONT_PATH, midi_debug_file
        ]
        subprocess.run(fs_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        print("📊 Berechne Chroma...")
        y, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
        chroma_raw = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_chroma=12)
        
        chroma_steps = chroma_raw.T 
        norms = np.linalg.norm(chroma_steps, axis=1, keepdims=True)
        chroma_l2 = chroma_steps / (norms + 1e-9)
        num_steps = chroma_l2.shape[0]

        with open(header_file, "w") as f:
            f.write(f"// Generiert aus {input_file}\n")
            f.write(f"// BPM: {bpm}, Instrument: {instrument_name}\n")
            f.write("#ifndef SCORE_DATA_H\n#define SCORE_DATA_H\n\n")
            f.write(f"const int num_pages = {len(page_indices)};\n")
            arr_content = ", ".join(map(str, page_indices)) if page_indices else ""
            f.write(f"const int page_end_indices[] = {{ {arr_content} }};\n\n")
            f.write(f"const int score_len = {num_steps};\n")
            f.write("const float score_chroma[][12] = {\n")
            for i in range(num_steps):
                f.write("  {")
                f.write(", ".join([f"{v:.4f}f" for v in chroma_l2[i]]))
                f.write("}")
                if i < num_steps - 1: f.write(",\n")
                else: f.write("\n")
            f.write("};\n\n#endif\n")

        print(f"\n✅ FERTIG! ScoreData.h erstellt.")

    except Exception as e:
        print(f"\n❌ Fehler: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--bpm", type=int, default=100)
    parser.add_argument("--instrument", type=str, default="violin")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print("Datei nicht gefunden!")
    else:
        musicxml_to_teensy(args.input_file, args.bpm, args.instrument)