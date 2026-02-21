# =============================================================================
# generate_score_data.py – MusicXML/PDF → ScoreData.npz
# =============================================================================
# Pipeline: Partitur → MIDI → FluidSynth (Audio) → librosa (Chroma) → .npz
#
# Nutzung:
#   python generate_score_data.py partitur.musicxml --bpm 40
#   python generate_score_data.py partitur.mxl --bpm 40 --instrument piano
#   python generate_score_data.py partitur.pdf --bpm 40   (braucht Audiveris)
#
# Benötigt: brew install fluidsynth + Soundfont in data/soundfonts/
# =============================================================================

import sys
import argparse
from pathlib import Path

from utils.chroma_builder import build_chroma
from utils.score_writer import write_score_data
from utils.omr import convert_pdf

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "generated"

MUSICXML_EXTENSIONS = {'.musicxml', '.mxl', '.xml'}
PDF_EXTENSIONS = {'.pdf'}


def main():
    parser = argparse.ArgumentParser(
        description="Generiert ScoreData.npz aus einer Partitur (PDF oder MusicXML)."
    )
    parser.add_argument("input_file", help="PDF, MusicXML oder MXL Datei")
    parser.add_argument("--bpm", type=int, default=40, help="Tempo in BPM (Standard: 40)")
    parser.add_argument("--instrument", type=str, default="violin",
                        help="Instrument (violin, piano, cello, flute, ... Standard: violin)")
    parser.add_argument("--output", type=str, default=None,
                        help="Ausgabedatei (Standard: data/generated/<name>.npz)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"FEHLER: Datei nicht gefunden: {input_path}")
        sys.exit(1)

    suffix = input_path.suffix.lower()

    # 1. PDF → MusicXML (falls nötig)
    if suffix in PDF_EXTENSIONS:
        print(f"\n[1/3] OMR: PDF → MusicXML")
        try:
            musicxml_path = convert_pdf(str(input_path))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"FEHLER: {e}")
            sys.exit(1)
    elif suffix in MUSICXML_EXTENSIONS:
        print(f"\n[1/3] MusicXML erkannt, OMR übersprungen")
        musicxml_path = str(input_path)
    else:
        print(f"FEHLER: Unbekanntes Format '{suffix}'. Unterstützt: PDF, MusicXML, MXL")
        sys.exit(1)

    # 2. MusicXML → MIDI → FluidSynth → Chroma + Seitenumbrüche
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"{input_path.stem}.npz"
    wav_output_path = output_path.with_suffix(".wav")

    print(f"\n[2/3] Synthetisiere Audio und berechne Chroma")
    try:
        chroma, page_indices = build_chroma(
            musicxml_path, args.bpm, args.instrument,
            wav_output_path=str(wav_output_path)
        )
    except Exception as e:
        print(f"FEHLER bei Chroma-Berechnung: {e}")
        sys.exit(1)

    # 3. ScoreData.npz speichern

    print(f"\n[3/3] Speichere {output_path}")
    metadata = f"Generiert aus {input_path.name}, BPM: {args.bpm}, Instrument: {args.instrument}"
    write_score_data(str(output_path), chroma, page_indices, metadata)

    print(f"\nFERTIG! {output_path}")
    print(f"  Frames: {chroma.shape[1]}, Seiten: {len(page_indices) + 1}")


if __name__ == "__main__":
    main()
