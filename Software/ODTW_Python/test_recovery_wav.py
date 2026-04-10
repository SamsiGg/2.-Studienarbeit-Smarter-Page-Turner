"""
test_recovery_wav.py – Recovery-Evaluation mit echter WAV-Aufnahme
===================================================================
Voraussetzung: WAV-Datei ist das gesamte Stück, aufgenommen bei exakt
dem gleichen BPM wie die Partitur → WAV-Frame i ≈ Score-Frame i * Skalierungsfaktor.

Methode pro Szenario:
  1. WARMUP: Tracker läuft auf WAV-Frames ab start_frac × N_wav (lockt ein)
  2. SPRUNG: Audio-Input springt zu jump_frac × N_wav
  3. Weiter bis Ende der WAV (oder MAX_EXTRA_FRAMES nach Recovery)

Erfolg wenn:
  a) Recovery wird ausgelöst
  b) Position nach Recovery liegt in ±TOLERANCE_TAKTE der erwarteten Score-Position
  c) Kein weiterer Recovery-Sprung bis zum Ende (= stabil durchgehalten)

Ausführung:
    python3 test_recovery_wav.py \\
        --score ../data/scores/Fiocco_Musescore.npz \\
        --wav   ../data/audio/Fiocco-Live-40bpm.wav
"""

import os
import sys
import time
import argparse
import numpy as np
import librosa
from collections import deque

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

WARMUP_FRAMES   = 400    # Frames korrekt tracken bevor gesprungen wird
TOLERANCE_TAKTE = 3.0    # Akzeptable Abweichung nach Recovery (Takte)
MAX_EXTRA_FRAMES = 2000  # Max. Frames nach Recovery bis Stabilitätscheck
FAKE_RMS        = 0.05   # Simulierter RMS-Pegel

LIVE_WAV_DEFAULT = os.path.join(os.path.dirname(__file__),
    "..", "data", "audio", "Fiocco-Live-40bpm.wav")

# Sprung-Szenarien: (start_frac, jump_to_frac, Label)
JUMP_SCENARIOS = [
    (0.05, 0.50, "Anfang  →  Mitte"),
    (0.05, 0.75, "Anfang  →  3/4"),
    (0.25, 0.75, "1/4     →  3/4"),
    (0.50, 0.10, "Mitte   →  1/10"),
    (0.50, 0.75, "Mitte   →  3/4"),
    (0.75, 0.25, "3/4     →  1/4"),
    (0.25, 0.50, "1/4     →  Mitte"),
]


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def frames_to_takte(n: float) -> float:
    return (n * HOP_LENGTH / SAMPLE_RATE) / ((60.0 / BPM) * BEATS_PER_MEASURE)


def takte_to_frames(t: float) -> int:
    return int(t * (60.0 / BPM) * BEATS_PER_MEASURE * SAMPLE_RATE / HOP_LENGTH)


def load_wav_chroma(wav_file: str) -> np.ndarray:
    """WAV → Chroma (12, N) mit Live-System-Einstellungen."""
    print(f'Lade WAV: {wav_file} …')
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr,
        n_fft=BLOCK_SIZE, hop_length=HOP_LENGTH,
        center=False, tuning=0, n_chroma=12,
    )
    return chroma   # (12, N_wav)


def make_tracker(ref_chroma: np.ndarray) -> ODTWTracker:
    return ODTWTracker(
        reference_chroma=ref_chroma,
        page_end_indices=[],
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

def calibrate_positions(ref_chroma: np.ndarray,
                        wav_chroma: np.ndarray) -> np.ndarray:
    """
    Einmalige Kalibrierung: Tracker läuft einmal komplett durch die WAV
    und liefert für jeden WAV-Frame die tatsächliche Score-Position.
    Wird verwendet um die "wahre" erwartete Score-Position am Sprungpunkt
    zu bestimmen – unabhängig von der Längen-Diskrepanz WAV/Score.
    """
    n_ref  = ref_chroma.shape[1]
    n_live = wav_chroma.shape[1]

    accumulated_costs = np.full(n_ref, np.inf)
    accumulated_costs[0] = 0.0
    current_position = 0
    chroma_buffer = deque(maxlen=max(1, SMOOTHING_WINDOW))
    positions = np.empty(n_live, dtype=int)

    for i in range(n_live):
        chroma_buffer.append(wav_chroma[:, i])
        avg  = np.mean(np.array(chroma_buffer), axis=0)
        norm = np.linalg.norm(avg)
        live_vec = avg / norm if norm > 1e-4 else avg

        start_scan = max(0, current_position - SEARCH_WINDOW)
        end_scan   = min(n_ref, current_position + SEARCH_WINDOW)
        new_costs  = np.full(n_ref, np.inf)
        best_cost  = np.inf
        best_index = current_position

        for j in range(start_scan, end_scan):
            dot  = max(-1.0, min(1.0, np.dot(live_vec, ref_chroma[:, j])))
            dist = 1.0 - dot
            c_wait = accumulated_costs[j]     + WAIT_PENALTY
            c_step = accumulated_costs[j - 1] + STEP_PENALTY if j > 0 else np.inf
            c_skip = accumulated_costs[j - 2] + SKIP_PENALTY if j > 1 else np.inf
            new_costs[j] = dist + min(c_wait, c_step, c_skip) * DAMPING_FACTOR
            if new_costs[j] < best_cost:
                best_cost  = new_costs[j]
                best_index = j

        accumulated_costs = new_costs
        current_position  = best_index
        positions[i]      = best_index

        if i % 2000 == 0:
            print(f'\r  Kalibrierung: {i}/{n_live} Frames …', end='', flush=True)

    print(f'\r  Kalibrierung abgeschlossen ({n_live} Frames).          ')
    return positions   # wav_frame → score_frame


def run_test(ref_chroma: np.ndarray, wav_chroma: np.ndarray,
             start_frac: float, jump_frac: float, label: str,
             calib_positions: np.ndarray) -> dict:
    """
    Simuliert Sprung mit echten WAV-Frames und misst Recovery.

    Die erwartete Score-Position am Sprungpunkt wird aus der Kalibrierung
    abgelesen (nicht linear interpoliert) – korrekt auch bei Wiederholungen
    oder Längen-Diskrepanz zwischen WAV und Score.
    """
    n_score = ref_chroma.shape[1]
    n_wav   = wav_chroma.shape[1]

    start_wav  = int(start_frac * n_wav)
    jump_wav   = int(jump_frac  * n_wav)
    # Wahre Score-Position am Sprungpunkt laut Kalibrierung
    jump_score = int(calib_positions[min(jump_wav, len(calib_positions) - 1)])

    tol_frames = takte_to_frames(TOLERANCE_TAKTE)

    tracker = make_tracker(ref_chroma)

    # Phase 1: Warmup ab start_wav
    warmup_end = min(start_wav + WARMUP_FRAMES, n_wav - 1)
    for i in range(start_wav, warmup_end):
        tracker.step(wav_chroma[:, i], FAKE_RMS)

    # Phase 2: Sprung – ab jump_wav bis Ende der WAV (oder MAX_EXTRA_FRAMES nach Recovery)
    recovered        = False
    frames_to_rec    = None
    pos_after_rec    = None
    extra_recoveries = 0
    reached_end      = False
    step_count       = 0

    for i in range(jump_wav, n_wav):
        state = tracker.step(wav_chroma[:, i], FAKE_RMS)
        step_count += 1

        if state.recovered:
            if not recovered:
                # Erste Recovery: Position messen
                recovered     = True
                frames_to_rec = step_count
                pos_after_rec = state.current_position
            else:
                extra_recoveries += 1

        # Nach Recovery: noch MAX_EXTRA_FRAMES weiterlaufen für Stabilitätscheck
        if recovered and step_count >= frames_to_rec + MAX_EXTRA_FRAMES:
            reached_end = True
            break

    if not recovered and step_count >= n_wav - jump_wav - 1:
        reached_end = True   # bis Ende gelaufen ohne Recovery

    # Bewertung
    error_frames = abs(pos_after_rec - jump_score) if pos_after_rec is not None else None
    error_takte  = frames_to_takte(error_frames) if error_frames is not None else None

    in_region  = error_takte is not None and error_takte <= TOLERANCE_TAKTE
    stable_end = extra_recoveries == 0 and (reached_end or not recovered)
    success    = recovered and in_region and stable_end

    return dict(
        success=success,
        recovered=recovered,
        frames_to_rec=frames_to_rec,
        pos_after_rec=pos_after_rec,
        jump_score=jump_score,
        error_frames=error_frames,
        error_takte=error_takte,
        in_region=in_region,
        extra_recoveries=extra_recoveries,
        stable_end=stable_end,
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
    """
    mid = ref_chroma.shape[1] // 2
    buf_size = RECOVERY_BUFFER_SIZE
    live_frames = ref_chroma[:, mid:mid + buf_size].T   # (buf_size, 12)
    norms = np.linalg.norm(live_frames, axis=1, keepdims=True)
    live_norm = np.where(norms > 1e-4, live_frames / norms, live_frames)

    step = 10
    ref_sub = ref_chroma[:, ::step]
    D = (1.0 - np.clip(live_norm @ ref_sub, -1.0, 1.0)).astype(np.float64)

    n_ref_full = ref_chroma.shape[1]
    m_sub      = ref_sub.shape[1]

    print(f'\n── Scan-Latenz-Benchmark ──────────────────────────────────────')
    print(f'   Live-Buffer: {buf_size} Frames  |  Referenz: {n_ref_full} Frames '
          f'(→ {m_sub} nach /10)  |  Wiederholungen: {n_reps}')

    # Numpy
    times_np = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _numpy_scan(D)
        times_np.append((time.perf_counter() - t0) * 1000)
    mean_np = sum(times_np) / len(times_np)
    print(f'   Numpy (kein Numba): Ø {mean_np:6.1f} ms  '
          f'(min {min(times_np):.1f}  max {max(times_np):.1f})')

    # Numba
    try:
        from dtw import _dtw_inner, _NUMBA_AVAILABLE
        if _NUMBA_AVAILABLE:
            _dtw_inner(D)   # Warmup / JIT-Kompilierung
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
              f'(Recovery würde {factor:.0f} Frames (~{mean_np / 1000 * SAMPLE_RATE / HOP_LENGTH:.0f} Frames) blockieren)')
    else:
        print(f'   → Numpy liegt im Budget.')
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Recovery-Evaluation mit echter WAV-Aufnahme')
    parser.add_argument('--score', default=SCORE_DATA_PATH)
    parser.add_argument('--wav',   default=LIVE_WAV_DEFAULT)
    args = parser.parse_args()

    # Laden
    print(f'Lade Score: {args.score}')
    archive    = np.load(args.score)
    ref_chroma = archive['chroma']
    n_score    = ref_chroma.shape[1]

    wav_chroma = load_wav_chroma(args.wav)
    n_wav      = wav_chroma.shape[1]

    print(f'\n  Score: {n_score} Frames ({frames_to_takte(n_score):.1f} T)')
    print(f'  WAV:   {n_wav} Frames ({frames_to_takte(n_wav):.1f} T)')
    print(f'  Längenverhältnis WAV/Score: {n_wav/n_score:.3f}x  '
          f'({"WAV länger" if n_wav > n_score else "Score länger"} → lineare Mapping wäre ungenau)')
    print(f'\n  Kalibrierung läuft (einmaliger Durchlauf ohne Recovery)...')
    calib_positions = calibrate_positions(ref_chroma, wav_chroma)

    print(f'\n  Recovery-Einstellungen:')
    print(f'    THRESHOLD={RECOVERY_THRESHOLD}, AVG_WINDOW={RECOVERY_AVG_WINDOW}, '
          f'BUFFER={RECOVERY_BUFFER_SIZE}, JUMP_THRESH={RECOVERY_JUMP_THRESHOLD}')
    print(f'    Toleranz: ±{TOLERANCE_TAKTE} Takte ({takte_to_frames(TOLERANCE_TAKTE)} Frames)')
    print(f'    Warmup: {WARMUP_FRAMES} Frames, Stabilitäts-Check: {MAX_EXTRA_FRAMES} Frames\n')

    print('=' * 80)
    print(f'  {"Szenario":<22}  {"Rec?":>4}  {"F→Rec":>6}  '
          f'{"Err[T]":>7}  {"Region":>6}  {"Stabil":>6}  {"":>4}')
    print(f'  {"-"*22}  {"-"*4}  {"-"*6}  {"-"*7}  {"-"*6}  {"-"*6}  {"-"*4}')

    results = []
    for start_frac, jump_frac, label in JUMP_SCENARIOS:
        # Genug WAV-Frames nach Sprung vorhanden?
        jump_wav = int(jump_frac * n_wav)
        if jump_wav + WARMUP_FRAMES + MAX_EXTRA_FRAMES > n_wav:
            print(f'  {label:<22}  übersprungen (zu nah am WAV-Ende)')
            continue
        # Genug Warmup-Frames vorhanden?
        start_wav = int(start_frac * n_wav)
        if start_wav + WARMUP_FRAMES > n_wav:
            print(f'  {label:<22}  übersprungen (Warmup-Bereich zu kurz)')
            continue

        r = run_test(ref_chroma, wav_chroma, start_frac, jump_frac, label, calib_positions)
        results.append((label, r))

        rec_str    = 'JA'   if r['recovered']    else 'NEIN'
        frec_str   = str(r['frames_to_rec'])      if r['frames_to_rec']  else '–'
        err_str    = f'{r["error_takte"]:.2f} T'  if r['error_takte'] is not None else '–'
        region_str = '✓'    if r['in_region']     else '✗'
        stabil_str = '✓'    if r['stable_end']    else f'✗ (+{r["extra_recoveries"]})'
        ok_str     = '✓'    if r['success']       else '✗'

        print(f'  {label:<22}  {rec_str:>4}  {frec_str:>6}  '
              f'{err_str:>7}  {region_str:>6}  {stabil_str:>6}  {ok_str:>4}')

    # Zusammenfassung
    n_tested  = len(results)
    if n_tested == 0:
        print('  Keine Szenarien auswertbar.')
        return

    n_rec     = sum(1 for _, r in results if r['recovered'])
    n_region  = sum(1 for _, r in results if r['in_region'])
    n_stable  = sum(1 for _, r in results if r['stable_end'] and r['recovered'])
    n_success = sum(1 for _, r in results if r['success'])

    avg_frec = None
    if n_rec > 0:
        vals = [r['frames_to_rec'] for _, r in results if r['frames_to_rec']]
        avg_frec = sum(vals) / len(vals)

    print('=' * 80)
    print(f'\nZusammenfassung ({n_tested} Szenarien):')
    print(f'  Recovery ausgelöst:        {n_rec:2d} / {n_tested}  '
          f'({100*n_rec/n_tested:.0f} %)')
    print(f'  Position korrekt (±{TOLERANCE_TAKTE}T):  {n_region:2d} / {n_tested}  '
          f'({100*n_region/n_tested:.0f} %)')
    print(f'  Danach stabil:             {n_stable:2d} / {n_tested}  '
          f'({100*n_stable/n_tested:.0f} %)')
    print(f'  Gesamt erfolgreich:        {n_success:2d} / {n_tested}  '
          f'({100*n_success/n_tested:.0f} %)')
    if avg_frec:
        print(f'  Ø Frames bis Recovery:     {avg_frec:.0f}  '
              f'({frames_to_takte(avg_frec):.2f} T,  '
              f'{avg_frec * HOP_LENGTH / SAMPLE_RATE:.1f} s)')
    print()

    benchmark_scan_latency(ref_chroma)


if __name__ == '__main__':
    main()
