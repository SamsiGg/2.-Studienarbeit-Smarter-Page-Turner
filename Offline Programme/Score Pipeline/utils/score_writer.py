# =============================================================================
# score_writer.py – Speichert ScoreData als .npz (NumPy-Archiv)
# =============================================================================

import numpy as np


def write_score_data(filepath: str, chroma: np.ndarray, page_end_indices: list[int],
                     metadata: str = ""):
    """Speichert Chroma-Daten als .npz Datei.

    Enthält:
        - chroma: Shape (12, N) – L2-normalisierte Chroma-Vektoren
        - page_end_indices: Frame-Indizes der Seitenenden
        - metadata: Beschreibungstext

    Args:
        filepath: Ausgabepfad (z.B. "ScoreData.npz").
        chroma: Shape (12, N) – L2-normalisierte Chroma-Vektoren.
        page_end_indices: Frame-Indizes der Seitenenden.
        metadata: Optionaler Beschreibungstext.
    """
    np.savez(filepath,
             chroma=chroma,
             page_end_indices=np.array(page_end_indices, dtype=np.int32),
             metadata=np.array(metadata))

    num_frames = chroma.shape[1]
    print(f"ScoreData gespeichert: {filepath}")
    print(f"  {num_frames} Frames, {len(page_end_indices)} Seitengrenzen")
