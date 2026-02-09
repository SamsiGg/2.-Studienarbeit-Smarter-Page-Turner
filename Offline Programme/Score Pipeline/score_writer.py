# =============================================================================
# score_writer.py – Schreibt ScoreData.h im Teensy-kompatiblen C-Header-Format
# =============================================================================

import numpy as np


def write_score_data(filepath: str, chroma: np.ndarray, page_end_indices: list[int],
                     metadata: str = ""):
    """Schreibt Chroma-Daten als C-Header-Datei (ScoreData.h).

    Format ist identisch mit dem Output von musescore_to_chroma.py,
    damit die gleiche Datei auf dem Teensy und in der Python-Version läuft.

    Args:
        filepath: Ausgabepfad (z.B. "ScoreData.h").
        chroma: Shape (12, N) – L2-normalisierte Chroma-Vektoren.
        page_end_indices: Frame-Indizes der Seitenenden.
        metadata: Optionaler Kommentar (z.B. "Generiert aus Fiocco.musicxml").
    """
    num_frames = chroma.shape[1]

    with open(filepath, "w") as f:
        # Header-Kommentar
        if metadata:
            f.write(f"// {metadata}\n")
        f.write("#ifndef SCORE_DATA_H\n#define SCORE_DATA_H\n\n")

        # Page-Turn-Indices
        f.write(f"const int num_pages = {len(page_end_indices)};\n")
        arr = ", ".join(map(str, page_end_indices)) if page_end_indices else ""
        f.write(f"const int page_end_indices[] = {{ {arr} }};\n\n")

        # Chroma-Daten
        f.write(f"const int score_len = {num_frames};\n")
        f.write("const float score_chroma[][12] = {\n")
        for i in range(num_frames):
            row = ", ".join(f"{v:.4f}f" for v in chroma[:, i])
            comma = "," if i < num_frames - 1 else ""
            f.write(f"  {{{row}}}{comma}\n")
        f.write("};\n\n#endif\n")

    print(f"ScoreData.h geschrieben: {num_frames} Frames, {len(page_end_indices)} Seitengrenzen")
