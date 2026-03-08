# =============================================================================
# score_loader.py – Parser für ScoreData (.npz oder .h)
# =============================================================================
# Unterstützt zwei Formate:
#   .npz  – NumPy-Archiv (aktuelles Format, von generate_score_data.py)
#   .h    – C-Header (altes Format, für Rückwärtskompatibilität)
#
# Teensy-Äquivalent: ScoreData.h wird dort direkt als C-Array eingebunden.
# =============================================================================

import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ScoreData:
    """Geparste Darstellung einer ScoreData-Datei."""
    num_pages: int                  # Anzahl Seiten
    page_end_indices: list[int]     # Frame-Indizes der Seitenenden
    score_len: int                  # Gesamtanzahl Chroma-Frames
    chroma: np.ndarray              # Shape (12, N) – Referenz-Chroma
    filepath: str                   # Quelldatei
    bpm: int = 40                   # Tempo in BPM (aus NPZ oder settings-Fallback)
    beats_per_measure: int = 4      # Schläge pro Takt
    musicxml_content: str = ""      # MusicXML-Quelltext (leer = nicht verfügbar)
    silence_mask: Optional[np.ndarray] = None  # (N,) bool – True = stiller Frame (zufälliges Chroma)


def load_score_data(filepath: str) -> ScoreData:
    """ScoreData laden – erkennt automatisch .npz oder .h Format."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ScoreData nicht gefunden: {filepath}")

    if path.suffix == '.npz':
        return _load_npz(path)
    else:
        return _load_header(path)


def _load_npz(path: Path) -> ScoreData:
    """Lädt ScoreData aus .npz (NumPy-Archiv)."""
    import settings
    print(f"Lade {path}...")
    data = np.load(str(path), allow_pickle=True)

    chroma = data['chroma']                                     # (12, N)
    page_end_indices = data['page_end_indices'].tolist()        # [int, ...]

    # BPM + Taktart – Fallback auf settings für ältere NPZ-Dateien ohne diese Felder
    bpm = int(data['bpm']) if 'bpm' in data else settings.BPM
    beats_per_measure = int(data['beats_per_measure']) if 'beats_per_measure' in data else settings.BEATS_PER_MEASURE
    musicxml_content = str(data['musicxml_content']) if 'musicxml_content' in data else ""

    # silence_mask: welche Frames haben zufälliges Chroma (Stille/Pausen)
    silence_mask = data['silence_mask'].astype(bool) if 'silence_mask' in data else None

    # Score am letzten echten Musik-Frame trimmen (silence_mask-basiert)
    # → verhindert, dass der Tracker bei Stille ans Ende driftet
    if silence_mask is not None:
        real_frames = np.where(~silence_mask)[0]
        if len(real_frames) > 0:
            trim_idx = int(real_frames[-1]) + 1
            if trim_idx < chroma.shape[1]:
                removed = chroma.shape[1] - trim_idx
                print(f"  Stille getrimmt: {chroma.shape[1]} → {trim_idx} Frames "
                      f"({removed} stille Frames am Ende entfernt)")
                chroma = chroma[:, :trim_idx]
                silence_mask = silence_mask[:trim_idx]
                page_end_indices = [i for i in page_end_indices if i < trim_idx]

    num_pages = len(page_end_indices) + 1

    print(f"  Seiten: {num_pages}, Seitengrenzen: {page_end_indices}")
    print(f"  Frames: {chroma.shape[1]}, Chroma-Shape: {chroma.shape}")
    print(f"  BPM: {bpm}, Taktart: {beats_per_measure}/4")
    print(f"  L2-Norm Frame 0: {np.linalg.norm(chroma[:, 0]):.4f}")

    return ScoreData(
        num_pages=num_pages,
        page_end_indices=[int(i) for i in page_end_indices],
        score_len=chroma.shape[1],
        chroma=chroma.astype(np.float32),
        filepath=str(path),
        bpm=bpm,
        beats_per_measure=beats_per_measure,
        musicxml_content=musicxml_content,
        silence_mask=silence_mask,
    )


def _load_header(path: Path) -> ScoreData:
    """Lädt ScoreData aus ScoreData.h (C-Header, Legacy-Format)."""
    print(f"Lade {path}...")
    content = path.read_text()

    num_pages = _parse_int(content, r'num_pages\s*=\s*(\d+)')
    score_len = _parse_int(content, r'score_len\s*=\s*(\d+)')
    page_end_indices = _parse_int_array(content, 'page_end_indices')
    chroma = _parse_chroma(content)

    if len(page_end_indices) != num_pages:
        print(f"WARNUNG: num_pages={num_pages}, aber {len(page_end_indices)} Indizes gefunden.")
    if chroma.shape[1] != score_len:
        print(f"WARNUNG: score_len={score_len}, aber {chroma.shape[1]} Frames geparst.")

    print(f"  Seiten: {num_pages}, Seitengrenzen: {page_end_indices}")
    print(f"  Frames: {chroma.shape[1]}, Chroma-Shape: {chroma.shape}")
    print(f"  L2-Norm Frame 0: {np.linalg.norm(chroma[:, 0]):.4f}")

    return ScoreData(
        num_pages=num_pages,
        page_end_indices=page_end_indices,
        score_len=chroma.shape[1],
        chroma=chroma,
        filepath=str(path),
    )


def _parse_int(content: str, pattern: str) -> int:
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Pattern nicht gefunden: {pattern}")
    return int(match.group(1))


def _parse_int_array(content: str, var_name: str) -> list[int]:
    pattern = rf'{var_name}\s*\[\s*\]\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Array '{var_name}' nicht gefunden.")
    return [int(x.strip()) for x in match.group(1).split(',') if x.strip()]


def _parse_chroma(content: str) -> np.ndarray:
    keyword = "score_chroma"
    start_pos = content.find(keyword)
    if start_pos == -1:
        raise ValueError(f"'{keyword}' nicht in der Datei gefunden!")

    content_chroma = content[start_pos:]
    array_start = content_chroma.find('{')
    array_end = content_chroma.rfind('}')
    data_string = content_chroma[array_start:array_end + 1]

    clean = data_string.replace('f', '').replace('{', '').replace('}', '').replace(';', '')
    tokens = clean.replace(',', ' ').split()

    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            continue

    if len(values) == 0:
        raise ValueError("Keine Chroma-Werte gefunden!")
    if len(values) % 12 != 0:
        print(f"WARNUNG: {len(values)} Werte nicht durch 12 teilbar. Schneide ab.")
        values = values[:(len(values) // 12) * 12]

    return np.array(values, dtype=np.float32).reshape(-1, 12).T
