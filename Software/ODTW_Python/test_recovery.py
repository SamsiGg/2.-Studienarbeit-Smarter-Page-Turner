"""
test_recovery.py – Evaluation der Recovery-Funktion (N4 / Robustheit)
======================================================================
Testet, ob ODTWTracker nach einem simulierten Sprung (Positionsfehler)
erfolgreich zur richtigen Stelle zurückfindet.

Methode:
  1. Tracker wird mit Score-Chroma als Live-Input gestartet (gute Bedingungen,
     kein Rauschen, Normaltempo → Tracker folgt korrekt).
  2. Nach WARMUP_FRAMES wird der Audio-Input auf eine andere Position
     des Stücks umgeschaltet (simulierter Sprung = Positionsfehler).
  3. Gemessen wird: Triggert Recovery? Wie schnell? Wie genau?

Metrik:
  Fehler = |tracker_position – wahre_position| in Frames nach Recovery.
  Erfolg wenn Fehler < TOLERANCE_FRAMES.

Ausführung:
    python3 test_recovery.py
    python3 test_recovery.py --score ../data/scores/Fiocco_Musescore.npz
"""

import os
import sys
import time
import argparse
import numpy as np

_LPT_DIR = os.path.join(os.path.dirname(__file__), "..", "Live Page Turner")
sys.path.insert(0, _LPT_DIR)

from settings import (
    SAMPLE_RATE, BLOCK_SIZE, HOP_LENGTH,
    SEARCH_WINDOW, DAMPING_FACTOR,
    WAIT_PENALTY, SKIP_PENALTY, STEP_PENALTY,
    SMOOTHING_WINDOW, BPM, BEATS_PER_MEASURE,
    START_THRESHOLD_RMS,
    RECOVERY_THRESHOLD, RECOVERY_AVG_WINDOW,
    RECOVERY_BUFFER_SIZE, RECOVERY_JUMP_THRESHOLD,
    SILENCE_FRAMES_BEFORE_SLEEP,
    SCORE_DATA_PATH,
)
from dtw import ODTWTracker

# ── Konfiguration ──────────────────────────────────────────────────────────────

WARMUP_FRAMES    = 300   # Frames korrekt tracken, bevor gesprungen wird
MAX_WAIT_FRAMES  = 1500  # Maximale Frames nach Sprung, bevor Test als Fehler gilt
TOLERANCE_FRAMES = 200   # Akzeptable Abweichung nach Recovery (in Frames)
FAKE_RMS         = 0.05  # Simulierter RMS-Pegel (über START_THRESHOLD_RMS)

# Sprung-Szenarien: (start_frac, jump_to_frac, Label)
# start_frac  = relative Position im Stück, an der getrackt wird (0.0 = Anfang)
# jump_to_frac = relative Zielposition, ab der Audio umgeschaltet wird
JUMP_SCENARIOS = [
    (0.05, 0.50, "Anfang  →  Mitte"),
    (0.05, 0.75, "Anfang  →  3/4"),
    (0.25, 0.75, "1/4     →  3/4"),
    (0.50, 0.10, "Mitte   →  Anfang"),
    (0.50, 0.75, "Mitte   →  3/4"),
    (0.75, 0.25, "3/4     →  1/4"),
    (0.75, 0.05, "3/4     →  Anfang"),
    (0.25, 0.50, "1/4     →  Mitte"),
]


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def frames_to_measures(n: float) -> float:
    return (n * HOP_LENGTH / SAMPLE_RATE) / ((60.0 / BPM) * BEATS_PER_MEASURE)


def make_tracker(ref_chroma: np.ndarray) -> ODTWTracker:
    return ODTWTracker(
        reference_chroma=ref_chroma,
        page_end_indices=[],           # Seitenwechsel für diesen Test irrelevant
        search_window=SEARCH_WINDOW,
        damping_factor=DAMPING_FACTOR,
        wait_penalty=WAIT_PENALTY,
        skip_penalty=SKIP_PENALTY,
        step_penalty=STEP_PENALTY,
        start_threshold_rms=START_THRESHOLD_RMS,
        smoothing_window=SMOOTHING_WINDOW,
        bpm=BPM,
        beats_per_measure=BEATS_PER_MEASURE,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE,
        use_recovery=True,
        recovery_threshold=RECOVERY_THRESHOLD,
        recovery_avg_window=RECOVERY_AVG_WINDOW,
        recovery_buffer_size=RECOVERY_BUFFER_SIZE,
        recovery_jump_threshold=RECOVERY_JUMP_THRESHOLD,
        silence_frames_before_sleep=SILENCE_FRAMES_BEFORE_SLEEP,
    )


# ── Einzeltest ─────────────────────────────────────────────────────────────────

def run_jump_test(ref_chroma: np.ndarray,
                  start_frac: float,
                  jump_frac:  float,
                  label:      str) -> dict:
    """
    Simuliert einen Positionssprung und misst ob Recovery greift.

    Returns dict mit:
      success        – Recovery hat Position korrekt gefunden
      recovered      – Recovery wurde überhaupt getriggert
      frames_to_rec  – Frames nach Sprung bis Recovery (None wenn kein Recovery)
      error_frames   – Fehler in Frames nach Recovery (None wenn kein Recovery)
      error_takte    – Fehler in Takten nach Recovery
      final_pos      – letzte bekannte Position des Trackers
      true_pos       – wahre Zielposition
    """
    n = ref_chroma.shape[1]
    start_pos = int(start_frac * n)
    jump_pos  = int(jump_frac  * n)

    tracker = make_tracker(ref_chroma)

    # Phase 1: Warmup – tracker bei start_pos einlaufen lassen
    for i in range(WARMUP_FRAMES):
        frame_idx = (start_pos + i) % n
        tracker.step(ref_chroma[:, frame_idx], FAKE_RMS)

    # Phase 2: Sprung – Audio von jump_pos einspeisen, auf Recovery warten
    frames_to_rec = None
    error_frames  = None
    recovered     = False
    final_pos     = tracker.current_position

    for i in range(MAX_WAIT_FRAMES):
        frame_idx = (jump_pos + i) % n
        state = tracker.step(ref_chroma[:, frame_idx], FAKE_RMS)
        final_pos = state.current_position

        if state.recovered and not recovered:
            recovered = True
            frames_to_rec = i + 1
            error_frames  = abs(final_pos - jump_pos)

        # Nach Recovery noch 50 Frames weiterlaufen damit Tracker einschwingt
        if recovered and i >= frames_to_rec + 50:
            error_frames = abs(final_pos - jump_pos)
            break

    success = recovered and error_frames is not None and error_frames < TOLERANCE_FRAMES

    return dict(
        success=success,
        recovered=recovered,
        frames_to_rec=frames_to_rec,
        error_frames=error_frames,
        error_takte=frames_to_measures(error_frames) if error_frames is not None else None,
        final_pos=final_pos,
        true_pos=jump_pos,
    )


# ── Scan-Latenz-Benchmark ──────────────────────────────────────────────────────

def _numpy_scan(D: np.ndarray) -> int:
    """Numpy-only Subsequence-DTW (identisch zum Fallback in dtw.py)."""
    n, m_sub = D.shape
    dp = np.full((n + 1, m_sub), np.inf)
    dp[0, :] = 0.0
    for i in range(1, n + 1):
        d_row = D[i - 1]
        prev_diag = np.empty(m_sub)
        prev_diag[0] = np.inf
        prev_diag[1:] = dp[i - 1, :-1]
        dp[i] = d_row + np.minimum(dp[i - 1], prev_diag)
        for j in range(1, m_sub):
            candidate = dp[i, j - 1] + d_row[j]
            if candidate < dp[i, j]:
                dp[i, j] = candidate
    return int(np.argmin(dp[n, :]))


def benchmark_scan_latency(ref_chroma: np.ndarray, n_reps: int = 5):
    """
    Misst die Latenz eines einzelnen Recovery-Scans:
      - Numpy-only (immer verfügbar)
      - Numba JIT (falls installiert)

    Verwendet RECOVERY_BUFFER_SIZE Live-Frames (wie im echten Recovery-Fall).
    Referenz wird auf jeden 10. Frame reduziert (wie in dtw._full_score_scan).
    """
    # Synthetische Live-Frames aus Partitur-Mitte
    mid = ref_chroma.shape[1] // 2
    buf_size = RECOVERY_BUFFER_SIZE
    live_frames = ref_chroma[:, mid:mid + buf_size].T   # (buf_size, 12)

    # Normalisieren
    norms = np.linalg.norm(live_frames, axis=1, keepdims=True)
    live_norm = np.where(norms > 1e-4, live_frames / norms, live_frames)

    step = 10
    ref_sub = ref_chroma[:, ::step]   # (12, m_sub)
    D = (1.0 - np.clip(live_norm @ ref_sub, -1.0, 1.0)).astype(np.float64)

    n_ref_full = ref_chroma.shape[1]
    m_sub      = ref_sub.shape[1]

    print(f'\n── Scan-Latenz-Benchmark ──────────────────────────────────────')
    print(f'   Live-Buffer: {buf_size} Frames  |  Referenz: {n_ref_full} Frames '
          f'(→ {m_sub} nach Subsampling /10)  |  Wiederholungen: {n_reps}')

    # Numpy-Benchmark
    times_np = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _numpy_scan(D)
        times_np.append((time.perf_counter() - t0) * 1000)
    mean_np = sum(times_np) / len(times_np)
    print(f'   Numpy (kein Numba): Ø {mean_np:6.1f} ms  '
          f'(min {min(times_np):.1f}  max {max(times_np):.1f})')

    # Numba-Benchmark (falls verfügbar)
    try:
        import sys as _sys
        _sys.path.insert(0, _LPT_DIR)
        from dtw import _dtw_inner, _NUMBA_AVAILABLE
        if _NUMBA_AVAILABLE:
            # Einmal warm laufen (JIT-Kompilierung)
            _dtw_inner(D)
            times_nb = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                _dtw_inner(D)
                times_nb.append((time.perf_counter() - t0) * 1000)
            mean_nb = sum(times_nb) / len(times_nb)
            speedup = mean_np / mean_nb if mean_nb > 0 else float('inf')
            print(f'   Numba (JIT):        Ø {mean_nb:6.1f} ms  '
                  f'(min {min(times_nb):.1f}  max {max(times_nb):.1f})  '
                  f'→ {speedup:.1f}× schneller')
        else:
            print('   Numba: nicht installiert.')
    except ImportError:
        print('   Numba: Import fehlgeschlagen.')

    budget_ms = HOP_LENGTH / SAMPLE_RATE * 1000
    print(f'   Audio-Budget pro Frame: {budget_ms:.1f} ms')
    if mean_np > budget_ms:
        factor = mean_np / budget_ms
        print(f'   → Numpy überschreitet Budget um {factor:.1f}× '
              f'(Recovery würde {mean_np / budget_ms:.0f} Frames blockieren)')
    else:
        print(f'   → Numpy liegt im Budget.')
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Recovery-Evaluation des ODTW-Trackers')
    parser.add_argument('--score', default=SCORE_DATA_PATH,
                        help='Partitur-Datei (.npz)')
    args = parser.parse_args()

    print(f'Lade Partitur: {args.score}')
    archive    = np.load(args.score)
    ref_chroma = archive['chroma']
    n          = ref_chroma.shape[1]
    total_m    = frames_to_measures(n)

    print(f'  {n} Frames  ({total_m:.1f} Takte)')
    print(f'\nRecovery-Einstellungen:')
    print(f'  RECOVERY_THRESHOLD    = {RECOVERY_THRESHOLD}')
    print(f'  RECOVERY_AVG_WINDOW   = {RECOVERY_AVG_WINDOW} Frames')
    print(f'  RECOVERY_BUFFER_SIZE  = {RECOVERY_BUFFER_SIZE} Frames')
    print(f'  RECOVERY_JUMP_THRESH  = {RECOVERY_JUMP_THRESHOLD}')
    print(f'  TOLERANCE             = {TOLERANCE_FRAMES} Frames '
          f'({frames_to_measures(TOLERANCE_FRAMES):.2f} Takte)')
    print(f'  WARMUP                = {WARMUP_FRAMES} Frames')
    print(f'  MAX_WAIT              = {MAX_WAIT_FRAMES} Frames\n')

    print('=' * 72)
    print(f'  {"Szenario":<22}  {"Rec?":>4}  {"F→Rec":>6}  '
          f'{"Fehler[F]":>9}  {"Fehler[T]":>9}  {"":>4}')
    print(f'  {"-"*22}  {"-"*4}  {"-"*6}  {"-"*9}  {"-"*9}  {"-"*4}')

    results = []
    for start_frac, jump_frac, label in JUMP_SCENARIOS:
        # Überspringen wenn Ziel außerhalb der Partitur
        if int(jump_frac * n) + MAX_WAIT_FRAMES > n:
            print(f'  {label:<22}  übersprungen (zu nah am Ende)')
            continue

        r = run_jump_test(ref_chroma, start_frac, jump_frac, label)
        results.append((label, r))

        rec_str   = 'JA' if r['recovered'] else 'NEIN'
        f_rec_str = f'{r["frames_to_rec"]}' if r['frames_to_rec'] else '–'
        err_f_str = f'{r["error_frames"]}' if r['error_frames'] is not None else '–'
        err_t_str = f'{r["error_takte"]:.2f}' if r['error_takte'] is not None else '–'
        ok_str    = '✓' if r['success'] else ('~' if r['recovered'] else '✗')

        print(f'  {label:<22}  {rec_str:>4}  {f_rec_str:>6}  '
              f'{err_f_str:>9}  {err_t_str:>9}  {ok_str:>4}')

    # Zusammenfassung
    n_tested  = len(results)
    n_rec     = sum(1 for _, r in results if r['recovered'])
    n_success = sum(1 for _, r in results if r['success'])

    avg_frames = None
    avg_takte  = None
    if n_rec > 0:
        valid_frames = [r['frames_to_rec'] for _, r in results if r['frames_to_rec']]
        avg_frames = sum(valid_frames) / len(valid_frames)
        avg_takte  = frames_to_measures(avg_frames)

    print('=' * 72)
    print(f'\nZusammenfassung:')
    print(f'  Szenarien getestet:   {n_tested}')
    print(f'  Recovery getriggert:  {n_rec} / {n_tested}  '
          f'({100*n_rec/n_tested:.0f} %)')
    print(f'  Erfolgreich (<{TOLERANCE_FRAMES} F):  {n_success} / {n_tested}  '
          f'({100*n_success/n_tested:.0f} %)')
    if avg_frames:
        print(f'  Ø Frames bis Recovery: {avg_frames:.0f}  '
              f'({avg_takte:.2f} Takte,  '
              f'{avg_frames * HOP_LENGTH / SAMPLE_RATE:.1f} s)')
    print()

    if n_rec < n_tested:
        print('  Hinweis: Recovery hat nicht in allen Fällen getriggert.')
        print(f'  RECOVERY_THRESHOLD ({RECOVERY_THRESHOLD}) evtl. zu hoch oder '
              f'MAX_WAIT_FRAMES ({MAX_WAIT_FRAMES}) zu niedrig.')

    benchmark_scan_latency(ref_chroma)


if __name__ == '__main__':
    main()
