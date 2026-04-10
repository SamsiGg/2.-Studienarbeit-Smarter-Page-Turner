"""
test_sanity.py – Schnelltest: Funktioniert ODTW überhaupt?
===========================================================
Testet den Algorithmus mit vier Szenarien:

  Test 1 – Selbst-Match:     Partitur als Live-Input → perfekte Diagonale erwartet
  Test 2 – Rauschen:         Partitur + Rauschen σ=0.3 → sollte noch folgen
  Test 3 – WAV (live):       WAV mit center=False, tuning=0  (aktuelle live-Einstellung)
  Test 4 – WAV (score):      WAV mit center=True, tuning=auto (wie Score-Pipeline gebaut wurde)

  Außerdem: Diagnoseplot mit Fehler-Verlauf und Ähnlichkeit,
            Stille-Check am WAV-Anfang.

Ausführung:
    python3 test_sanity.py
    python3 test_sanity.py --score ../data/scores/Fiocco_Musescore.npz
"""

import os
import sys
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

_LPT_DIR = os.path.join(os.path.dirname(__file__), "..", "Live Page Turner")
sys.path.insert(0, _LPT_DIR)
from settings import (
    SAMPLE_RATE, BLOCK_SIZE, HOP_LENGTH,
    SEARCH_WINDOW, DAMPING_FACTOR,
    WAIT_PENALTY, SKIP_PENALTY, STEP_PENALTY,
    SMOOTHING_WINDOW, BPM, BEATS_PER_MEASURE,
    SCORE_DATA_PATH,
)

OUT_FILE = os.path.join(os.path.dirname(__file__),
    "..", "data", "generated", "sanity_check.png")
LIVE_WAV = os.path.join(os.path.dirname(__file__),
    "..", "data", "audio", "Pachelbel-Live-35bpm.wav")


# ── ODTW-Kern ─────────────────────────────────────────────────────────────────

def run_odtw(ref_chroma: np.ndarray, live_chroma: np.ndarray) -> np.ndarray:
    n_ref  = ref_chroma.shape[1]
    n_live = live_chroma.shape[1]

    accumulated_costs = np.full(n_ref, np.inf)
    accumulated_costs[0] = 0.0
    current_position = 0
    chroma_buffer = deque(maxlen=max(1, SMOOTHING_WINDOW))
    positions = np.empty(n_live, dtype=int)

    for i in range(n_live):
        chroma_buffer.append(live_chroma[:, i])
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

    return positions


def frames_to_measures(n):
    return (n * HOP_LENGTH / SAMPLE_RATE) / ((60.0 / BPM) * BEATS_PER_MEASURE)


def compute_mae(positions, ref_len):
    n     = len(positions)
    ideal = np.linspace(0, ref_len - 1, n)
    err   = np.abs(positions.astype(float) - ideal)
    return frames_to_measures(int(np.mean(err)))


def compute_similarity(ref_chroma, positions, live_chroma):
    """Mittlere Kosinus-Ähnlichkeit zwischen Live und Referenz an Tracking-Position."""
    sims = []
    n_ref = ref_chroma.shape[1]
    for i, p in enumerate(positions):
        p = min(int(p), n_ref - 1)
        live_vec = live_chroma[:, i]
        norm = np.linalg.norm(live_vec)
        if norm > 1e-4:
            live_vec = live_vec / norm
        sims.append(np.dot(live_vec, ref_chroma[:, p]))
    return float(np.mean(sims))


# ── WAV laden ─────────────────────────────────────────────────────────────────

def load_wav_chroma_live(wav_file):
    """center=False, tuning=0 – aktuelle Live-System-Einstellung."""
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=BLOCK_SIZE, hop_length=HOP_LENGTH,
        center=False, tuning=0, n_chroma=12
    ), y


def load_wav_chroma_score(wav_file):
    """center=True (default), tuning=auto – so wurde die Score-Chroma gebaut."""
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=BLOCK_SIZE, hop_length=HOP_LENGTH,
        n_chroma=12
        # center=True (default), tuning=None (auto) – wie chroma_builder.py
    )


# ── Stille-Check ──────────────────────────────────────────────────────────────

def silence_check(y, label):
    block = HOP_LENGTH
    n_blocks = len(y) // block
    rms_vals = [np.sqrt(np.mean(y[i*block:(i+1)*block]**2)) for i in range(min(n_blocks, 300))]
    threshold = 0.01
    silent = sum(1 for r in rms_vals if r < threshold)
    print(f'  Stille-Check ({label}): erste 300 Frames, '
          f'{silent} unter RMS={threshold} → '
          f'{"⚠ viel Stille am Anfang" if silent > 50 else "OK"}')
    return silent, rms_vals


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_path(ax, positions, ref_len, title, color, mae, sim):
    n     = len(positions)
    x     = np.linspace(0, frames_to_measures(ref_len), n)
    y_pos = np.array([frames_to_measures(p) for p in positions])
    total = frames_to_measures(ref_len)

    ax.plot([0, total], [0, total], color='#aab7b8', linewidth=1.5,
            linestyle='--', label='Ideal')
    ax.plot(x, y_pos, color=color, linewidth=1.5, label='Tracking')
    ax.set_xlim(0, total)
    ax.set_ylim(0, total * 1.05)
    ax.set_xlabel('Eingabe [T]', fontsize=9)
    ax.set_ylabel('Partitur [T]', fontsize=9)

    ok = mae < 1.0
    ax.set_title(f'{title}\nMAE={mae:.2f} T  sim={sim:.3f}', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#27ae60' if ok else '#c0392b')
        spine.set_linewidth(2.5)
    return ok


def plot_error(ax, positions, ref_len, title, color):
    n     = len(positions)
    ideal = np.linspace(0, ref_len - 1, n)
    err_t = np.array([frames_to_measures(abs(int(p - i)))
                      for p, i in zip(positions, ideal)])
    x     = np.linspace(0, frames_to_measures(ref_len), n)
    ax.plot(x, err_t, color=color, linewidth=1.2)
    ax.axhline(1.0, color='#888', linewidth=1.0, linestyle='--', label='1-Takt-Schwelle')
    ax.set_xlabel('Eingabe [T]', fontsize=9)
    ax.set_ylabel('Fehler [T]', fontsize=9)
    ax.set_title(f'Tracking-Fehler: {title}', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', default=SCORE_DATA_PATH)
    parser.add_argument('--wav',   default=LIVE_WAV)
    args = parser.parse_args()

    archive    = np.load(args.score)
    ref_chroma = archive['chroma']
    ref_len    = ref_chroma.shape[1]

    print(f'Partitur:  {ref_len} Frames ({frames_to_measures(ref_len):.1f} T)')
    print(f'Settings:  SEARCH_WINDOW={SEARCH_WINDOW}, DAMPING={DAMPING_FACTOR}, '
          f'WAIT={WAIT_PENALTY}, SKIP={SKIP_PENALTY}, BPM={BPM}')

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('ODTW Sanity Check', fontsize=13, fontweight='bold')

    # 2 Zeilen: oben Pfade, unten Fehler-Verläufe
    ax = [fig.add_subplot(2, 4, i+1) for i in range(8)]
    results = []

    # ── Test 1: Selbst-Match ──────────────────────────────────────────────────
    print('\nTest 1: Selbst-Match')
    pos1 = run_odtw(ref_chroma, ref_chroma.copy())
    mae1 = compute_mae(pos1, ref_len)
    sim1 = compute_similarity(ref_chroma, pos1, ref_chroma)
    ok1  = plot_path(ax[0], pos1, ref_len, 'T1: Selbst-Match', '#2e86c1', mae1, sim1)
    plot_error(ax[4], pos1, ref_len, 'T1: Selbst-Match', '#2e86c1')
    results.append(('Selbst-Match', mae1, ok1, sim1))
    print(f'  MAE={mae1:.3f} T  sim={sim1:.3f}  →  {"OK ✓" if ok1 else "FEHLER ✗"}')

    # ── Test 2: Rauschen σ=0.3 ───────────────────────────────────────────────
    print('\nTest 2: Rauschen σ=0.3')
    noisy = ref_chroma.copy() + np.random.normal(0, 0.3, ref_chroma.shape)
    noisy = np.clip(noisy, 0, None)
    norms = np.linalg.norm(noisy, axis=0); norms[norms == 0] = 1.0
    noisy /= norms
    pos2 = run_odtw(ref_chroma, noisy)
    mae2 = compute_mae(pos2, ref_len)
    sim2 = compute_similarity(ref_chroma, pos2, ref_chroma)
    ok2  = plot_path(ax[1], pos2, ref_len, 'T2: Rauschen σ=0.3', '#8e44ad', mae2, sim2)
    plot_error(ax[5], pos2, ref_len, 'T2: Rauschen σ=0.3', '#8e44ad')
    results.append(('Rauschen σ=0.3', mae2, ok2, sim2))
    print(f'  MAE={mae2:.3f} T  sim={sim2:.3f}  →  {"OK ✓" if ok2 else "FEHLER ✗"}')

    if not os.path.isfile(args.wav):
        print(f'\nWAV nicht gefunden: {args.wav}')
        for a in [ax[2], ax[3], ax[6], ax[7]]:
            a.text(0.5, 0.5, 'WAV nicht gefunden', ha='center', va='center',
                   transform=a.transAxes, color='#888')
    else:
        # ── Stille-Check ──────────────────────────────────────────────────────
        print(f'\nLade WAV: {os.path.basename(args.wav)}')
        live_chroma_live, y_audio = load_wav_chroma_live(args.wav)
        live_chroma_score         = load_wav_chroma_score(args.wav)
        n_live = live_chroma_live.shape[1]
        print(f'  {n_live} Frames ({frames_to_measures(n_live):.1f} T)')
        silence_check(y_audio, 'WAV')

        # Cosinus-Ähnlichkeit der rohen Chromas: erste 200 Frames
        n_diag = min(200, ref_len, n_live)
        sim_live_raw  = float(np.mean([
            np.dot(live_chroma_live[:, i] / (np.linalg.norm(live_chroma_live[:, i]) + 1e-9),
                   ref_chroma[:, i]      / (np.linalg.norm(ref_chroma[:, i]) + 1e-9))
            for i in range(n_diag)]))
        sim_score_raw = float(np.mean([
            np.dot(live_chroma_score[:, i] / (np.linalg.norm(live_chroma_score[:, i]) + 1e-9),
                   ref_chroma[:, i]        / (np.linalg.norm(ref_chroma[:, i]) + 1e-9))
            for i in range(n_diag)]))
        print(f'\n  Feature-Vergleich (erste {n_diag} Frames, Diagonale):')
        print(f'  Ø Kosinus-Sim  live-Extraktion  (center=False, tuning=0):   {sim_live_raw:.3f}')
        print(f'  Ø Kosinus-Sim  score-Extraktion (center=True,  tuning=auto): {sim_score_raw:.3f}')
        print(f'  → Höherer Wert = besser passende Extraktion')

        # ── Test 3: WAV (live-Einstellung) ────────────────────────────────────
        print('\nTest 3: WAV (center=False, tuning=0 – live-Einstellung)')
        pos3 = run_odtw(ref_chroma, live_chroma_live)
        mae3 = compute_mae(pos3, ref_len)
        sim3 = compute_similarity(ref_chroma, pos3, live_chroma_live)
        ok3  = plot_path(ax[2], pos3, ref_len,
                         'T3: WAV (live)\ncenter=False tuning=0',
                         '#e67e22', mae3, sim3)
        plot_error(ax[6], pos3, ref_len, 'T3: WAV (live)', '#e67e22')
        results.append(('WAV live-Extraktion', mae3, ok3, sim3))
        print(f'  MAE={mae3:.3f} T  sim={sim3:.3f}  →  {"OK ✓" if ok3 else "FEHLER ✗"}')

        # ── Test 4: WAV (score-Einstellung) ───────────────────────────────────
        print('\nTest 4: WAV (center=True, tuning=auto – score-Einstellung)')
        pos4 = run_odtw(ref_chroma, live_chroma_score)
        mae4 = compute_mae(pos4, ref_len)
        sim4 = compute_similarity(ref_chroma, pos4, live_chroma_score)
        ok4  = plot_path(ax[3], pos4, ref_len,
                         'T4: WAV (score)\ncenter=True tuning=auto',
                         '#27ae60', mae4, sim4)
        plot_error(ax[7], pos4, ref_len, 'T4: WAV (score)', '#27ae60')
        results.append(('WAV score-Extraktion', mae4, ok4, sim4))
        print(f'  MAE={mae4:.3f} T  sim={sim4:.3f}  →  {"OK ✓" if ok4 else "FEHLER ✗"}')

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print('\n' + '='*60)
    all_ok = all(ok for _, _, ok, _ in results)
    print('  Gesamtergebnis:', '✓ BESTANDEN' if all_ok else '✗ FEHLER')
    print(f'  {"Test":<25}  {"MAE [T]":>8}  {"sim":>6}  {"":>4}')
    print(f'  {"-"*25}  {"-"*8}  {"-"*6}')
    for name, mae, ok, sim in results:
        print(f'  {name:<25}  {mae:>8.3f}  {sim:>6.3f}  {"✓" if ok else "✗"}')
    print('  (Schwelle: MAE < 1.0 Takt)')
    print('='*60)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches='tight', dpi=200)
    print(f'\nGespeichert: {OUT_FILE}')


if __name__ == '__main__':
    main()
