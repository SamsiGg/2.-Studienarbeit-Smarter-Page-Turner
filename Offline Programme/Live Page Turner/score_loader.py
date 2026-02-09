# =============================================================================
# score_loader.py – Parser für ScoreData.h
# =============================================================================
# Liest die C-Header-Datei und extrahiert:
#   - num_pages, page_end_indices[], score_len, score_chroma[][12]
#
# Teensy-Äquivalent: ScoreData.h wird dort direkt als C-Array eingebunden.
# Hier parsen wir die gleiche Datei zur Laufzeit.
# =============================================================================

import numpy as np
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoreData:
    """Geparste Darstellung einer ScoreData.h Datei."""
    num_pages: int                  # Anzahl Seiten
    page_end_indices: list[int]     # Frame-Indizes der Seitenenden
    score_len: int                  # Gesamtanzahl Chroma-Frames
    chroma: np.ndarray              # Shape (12, N) – Referenz-Chroma
    filepath: str                   # Quelldatei


def load_score_data(filepath: str) -> ScoreData:
    """ScoreData.h parsen und als ScoreData-Objekt zurückgeben."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"ScoreData nicht gefunden: {filepath}")

    print(f"Lade {filepath}...")
    content = path.read_text()

    # --- Metadaten extrahieren ---
    num_pages = _parse_int(content, r'num_pages\s*=\s*(\d+)')
    score_len = _parse_int(content, r'score_len\s*=\s*(\d+)')
    page_end_indices = _parse_int_array(content, 'page_end_indices')

    # --- Chroma-Daten extrahieren ---
    chroma = _parse_chroma(content)

    # --- Validierung ---
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
    """Einzelnen Integer per Regex extrahieren."""
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Pattern nicht gefunden: {pattern}")
    return int(match.group(1))


def _parse_int_array(content: str, var_name: str) -> list[int]:
    """Integer-Array aus C-Syntax extrahieren: const int name[] = { 1, 2, 3 };"""
    pattern = rf'{var_name}\s*\[\s*\]\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Array '{var_name}' nicht gefunden.")
    return [int(x.strip()) for x in match.group(1).split(',') if x.strip()]


def _parse_chroma(content: str) -> np.ndarray:
    """Chroma-Float-Array aus C-Syntax parsen. Gibt (12, N) zurück."""
    # Alles ab "score_chroma" finden
    keyword = "score_chroma"
    start_pos = content.find(keyword)
    if start_pos == -1:
        raise ValueError(f"'{keyword}' nicht in der Datei gefunden!")

    content_chroma = content[start_pos:]
    array_start = content_chroma.find('{')
    array_end = content_chroma.rfind('}')
    data_string = content_chroma[array_start:array_end + 1]

    # Bereinigen: f-Suffix, Klammern, Semikolons entfernen
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

    # Shape (N, 12) → Transponieren zu (12, N)
    return np.array(values, dtype=np.float32).reshape(-1, 12).T
