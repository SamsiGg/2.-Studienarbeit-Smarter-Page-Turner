import os
import sys
import argparse
import subprocess
import music21
# from midi2audio import FluidSynth  <-- Brauchen wir nicht mehr, macht nur Ärger
import librosa
import numpy as np

# --- KONFIGURATION ---
# Falls die .sf2 Datei direkt neben dem Skript liegt, trage nur den Namen ein:
SOUNDFONT_PATH = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/MuseScore_General.sf2" 
SAMPLE_RATE = 44100
FFT_SIZE = 4096
HOP_LENGTH = 2048

# Instrumenten-Mapping
INSTRUMENTS = {
    "piano": 0, "violin": 40, "viola": 41, "cello": 42, 
    "contrabass": 43, "flute": 73, "clarinet": 71, "trumpet": 56, "guitar": 24
}

# --- FIX FÜR MAC ---
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

def get_page_turn_indices(musicxml_path, sr, hop_length, bpm):
    """ Berechnet Frame-Indices für Seitenwechsel """
    print(f"📖 Analysiere Seitenumbrüche (Basis: {bpm} BPM)...")
    try:
        score = music21.converter.parse(musicxml_path)
        if not score.parts: return []
        part = score.parts[0]
        
        page_indices = []
        last_page = 1
        frames_per_beat = (60.0 / bpm) * (sr / hop_length)

        for measure in part.getElementsByClass('Measure'):
            try:
                current_page = 0
                if measure.pageNumber:
                    current_page = int(measure.pageNumber)
                if current_page > last_page:
                    offset_beats = measure.offset
                    frame_index = int(offset_beats * frames_per_beat)
                    print(f"   -> Seite {last_page} zu Ende bei Takt {measure.number} (Frame {frame_index})")
                    page_indices.append(frame_index)
                    last_page = current_page
            except: continue
        return page_indices
    except Exception as e:
        print(f"⚠️ Warnung Seitenumbrüche: {e}")
        return []

def pdf_to_teensy_header(input_file, bpm, instrument_name):
    base_name = os.path.splitext(input_file)[0]
    musicxml_file = base_name + ".musicxml"
    midi_debug_file = base_name + "_CHECK_THIS.mid" 
    wav_file = base_name + ".wav"
    header_file = "ScoreData.h"

    midi_program = INSTRUMENTS.get(instrument_name.lower(), 40)

    print(f"🚀 Starte Pipeline für: {input_file}")
    print(f"⚙️  Einstellungen: {bpm} BPM | Instrument: {instrument_name}")
    print("------------------------------------------------")
    
    # 1. OMR
    print("👁️  [1/4] OMR: Lese Noten...")
    patch_cmd = (
        "import sys; import numpy as np; np.int = int; "
        "from oemer import ete; "
        f"sys.argv = ['oemer', '{input_file}']; ete.main()"
    )
    
    try:
        if not os.path.exists(musicxml_file):
            subprocess.run([sys.executable, "-c", patch_cmd], check=True)
            possible_xml = input_file.rsplit(".", 1)[0] + ".musicxml"
            if os.path.exists(possible_xml): os.rename(possible_xml, musicxml_file)
            elif os.path.exists(base_name + "/result.musicxml"): os.rename(base_name + "/result.musicxml", musicxml_file)
            
            if not os.path.exists(musicxml_file):
                # Fallback
                files = [f for f in os.listdir('.') if f.endswith('.musicxml')]
                if files: os.rename(max(files, key=os.path.getctime), musicxml_file)
    except Exception as e:
        print(f"❌ OMR Fehler: {e}")
        return

    # 2. Seitenwechsel
    page_turn_indices = get_page_turn_indices(musicxml_file, SAMPLE_RATE, HOP_LENGTH, bpm)

    # 3. MIDI Erstellen
    print(f"🎹 [2/4] Erstelle MIDI...")
    try:
        score = music21.converter.parse(musicxml_file)
        part = score.parts[0]
        
        # Tempo
        for m in part.getElementsByClass('Measure'): m.removeByClass('MetronomeMark')
        mm = music21.tempo.MetronomeMark(number=bpm)
        part.measures(1, 1).insert(0, mm)

        # Instrument
        part.removeByClass('Instrument')
        new_inst = music21.instrument.Instrument()
        new_inst.midiProgram = midi_program
        part.insert(0, new_inst)
        
        score.write('midi', fp=midi_debug_file)
    except Exception as e:
        print(f"❌ Fehler XML->MIDI: {e}")
        return

    # 4. Synthese (Manuell via Subprocess statt Library)
    print("\n🎻 [3/4] Synthetisiere Audio (Direct FluidSynth)...")
    
    # Check Soundfont
    if not os.path.exists(SOUNDFONT_PATH):
        print(f"🛑 FEHLER: Soundfont Datei nicht gefunden!")
        print(f"   Erwartet: {os.path.abspath(SOUNDFONT_PATH)}")
        print("   Bitte Pfad oben im Skript anpassen oder Datei hierher kopieren.")
        return

    try:
        # Der direkte Befehl für FluidSynth (Mac-kompatibel)
        # flags: -ni (no interface), -F (File export), -r (Rate), -g (Gain)
        fs_cmd = [
            "fluidsynth",
            "-ni", 
            "-g", "1.0",
            "-F", wav_file,
            "-r", str(SAMPLE_RATE),
            SOUNDFONT_PATH,
            midi_debug_file
        ]
        
        # Ausführen und Output unterdrücken (außer Fehler)
        subprocess.run(fs_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Synth Fehler (FluidSynth Code {e.returncode}):")
        print(e.stderr.decode()) # Zeige den echten Fehler von FluidSynth
        return
    except FileNotFoundError:
        print("❌ 'fluidsynth' Befehl nicht gefunden. Ist es installiert (brew install fluidsynth)?")
        return

    # 5. Chroma Analyse
    print("\n📊 [4/4] Erstelle Teensy Header...")
    try:
        # Warnung: audioread backend Fehler abfangen
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
            
            f.write(f"const int num_pages = {len(page_turn_indices)};\n")
            if len(page_turn_indices) > 0:
                f.write("const int page_end_indices[] = { " + ", ".join(map(str, page_turn_indices)) + " };\n\n")
            else:
                f.write("const int page_end_indices[] = { };\n\n")

            f.write(f"const int score_len = {num_steps};\n")
            f.write("const float score_chroma[][12] = {\n")
            for i in range(num_steps):
                f.write("  {")
                f.write(", ".join([f"{v:.4f}f" for v in chroma_l2[i]]))
                f.write("}")
                if i < num_steps - 1: f.write(",\n")
                else: f.write("\n")
            f.write("};\n\n#endif\n")
            
    except Exception as e:
        print(f"❌ Audio Fehler: {e}")
        return

    print(f"\n✅ FERTIG! ScoreData.h erstellt. Prüfe die MIDI Datei '{midi_debug_file}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--bpm", type=int, default=100)
    parser.add_argument("--instrument", type=str, default="violin")
    args = parser.parse_args()
    
    pdf_to_teensy_header(args.input_file, args.bpm, args.instrument)