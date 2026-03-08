# =============================================================================
# score_writer.py – Speichert ScoreData als .npz (NumPy-Archiv)
# =============================================================================

import numpy as np


def write_score_data(filepath: str, chroma: np.ndarray, page_end_indices: list[int],
                     metadata: str = "", bpm: int = 40, beats_per_measure: int = 4,
                     musicxml_content: str = "", silence_mask: np.ndarray = None):
    """Speichert Chroma-Daten als .npz Datei.

    Enthält:
        - chroma: Shape (12, N) – L2-normalisierte Chroma-Vektoren
        - page_end_indices: Frame-Indizes der Seitenenden
        - silence_mask: Boolean-Array (N,) – True = stiller Frame (zufälliges Chroma)
        - bpm: Tempo in BPM
        - beats_per_measure: Taktart (Zähler, z.B. 4 für 4/4)
        - metadata: Beschreibungstext

    Args:
        filepath: Ausgabepfad (z.B. "ScoreData.npz").
        chroma: Shape (12, N) – L2-normalisierte Chroma-Vektoren.
        page_end_indices: Frame-Indizes der Seitenenden.
        metadata: Optionaler Beschreibungstext.
        bpm: Tempo in BPM.
        beats_per_measure: Anzahl Schläge pro Takt.
        silence_mask: Boolean-Array (N,), True = stiller Frame. None → nicht gespeichert.
    """
    arrays = dict(
        chroma=chroma,
        page_end_indices=np.array(page_end_indices, dtype=np.int32),
        bpm=np.array(bpm, dtype=np.int32),
        beats_per_measure=np.array(beats_per_measure, dtype=np.int32),
        musicxml_content=np.array(musicxml_content),
        metadata=np.array(metadata),
    )
    if silence_mask is not None:
        arrays['silence_mask'] = np.array(silence_mask, dtype=bool)

    np.savez(filepath, **arrays)

    num_frames = chroma.shape[1]
    print(f"ScoreData gespeichert: {filepath}")
    print(f"  {num_frames} Frames, {len(page_end_indices)} Seitengrenzen, {bpm} BPM, {beats_per_measure}/4 Takt")
