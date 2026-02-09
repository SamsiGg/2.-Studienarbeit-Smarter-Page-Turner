# =============================================================================
# generate_score_data.py – PDF/MusicXML → ScoreData.h
# =============================================================================
# Einfache Pipeline: Partitur rein → ScoreData.h raus.
#
# Nutzung:
#   python generate_score_data.py partitur.musicxml --bpm 40
#   python generate_score_data.py partitur.mxl --bpm 40
#   python generate_score_data.py partitur.pdf --bpm 40   (braucht Audiveris)
#
# Keine Audio-Synthese nötig. Chroma wird direkt aus den Noten berechnet.
# =============================================================================

import sys
import argparse
from pathlib import Path

from chroma_builder import build_chroma
from score_writer import write_score_data
from omr import convert_pdf


MUSICXML_EXTENSIONS = {'.musicxml', '.mxl', '.xml'}
PDF_EXTENSIONS = {'.pdf'}


def main():
    parser = argparse.ArgumentParser(
        description="Generiert ScoreData.h aus einer Partitur (PDF oder MusicXML)."
    )
    parser.add_argument("input_file", help="PDF, MusicXML oder MXL Datei")
    parser.add_argument("--bpm", type=int, default=40, help="Tempo in BPM (Standard: 40)")
    parser.add_argument("--output", type=str, default="ScoreData.h",
                        help="Ausgabedatei (Standard: ScoreData.h)")
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

    # 2. MusicXML → Chroma + Seitenumbrüche
    print(f"\n[2/3] Berechne Chroma-Vektoren aus Noten")
    try:
        chroma, page_indices = build_chroma(musicxml_path, args.bpm)
    except Exception as e:
        print(f"FEHLER bei Chroma-Berechnung: {e}")
        sys.exit(1)

    # 3. ScoreData.h schreiben
    print(f"\n[3/3] Schreibe {args.output}")
    metadata = f"Generiert aus {input_path.name}, BPM: {args.bpm}"
    write_score_data(args.output, chroma, page_indices, metadata)

    print(f"\nFERTIG! {args.output} erstellt.")
    print(f"  Frames: {chroma.shape[1]}, Seiten: {len(page_indices) + 1}")


if __name__ == "__main__":
    main()
