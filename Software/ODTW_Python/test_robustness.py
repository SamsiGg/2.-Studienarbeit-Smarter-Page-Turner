"""
ODTW Robustheit – Wissenschaftliche Evaluation (N2 & N3)
=========================================================
Ausführung:
    python3 test_robustness.py --score path/to/score.npz --wav path/to/live.wav

Ausgabe:
    robustness_tempo.pdf / .png   (N3 – Tempoinvarianz)
    robustness_noise.pdf / .png   (N2 – Rauschrobustheit)

Metriken:
    MAE      – Mean Absolute Error in Takten
    RMSE     – Root Mean Squared Error in Takten
    Acc@1T   – Anteil Frames mit Fehler ≤ 1 Takt

Hinweis: Verwendet exakt dieselben Parameter wie der Live Page Turner
         (settings.py), aber ohne Recovery und Sleep-Logik, um die
         Grundrobustheit des ODTW-Kerns isoliert zu messen.

Author: Samuel Geffert
"""

import os
import sys
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import deque
from scipy.ndimage import zoom

# ── Live-Page-Turner-Einstellungen importieren ────────────────────────────────
# Setzt den Suchpfad so, dass settings.py aus dem Live-Page-Turner-Ordner
# geladen wird – damit sind alle Parameter identisch zur Produktionslösung.
_LPT_DIR = os.path.join(os.path.dirname(__file__), "..", "Live Page Turner")
sys.path.insert(0, _LPT_DIR)

from settings import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    HOP_LENGTH,
    SEARCH_WINDOW,
    DAMPING_FACTOR,
    WAIT_PENALTY,
    SKIP_PENALTY,
    STEP_PENALTY,
    SMOOTHING_WINDOW,
    BPM,
    BEATS_PER_MEASURE,
    SCORE_DATA_PATH,
)

# ── Ausgabe-Verzeichnis ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

# Standard-WAV (kann per --wav überschrieben werden)
LIVE_WAV_FILE = os.path.join(os.path.dirname(__file__),
    "..", "data", "audio", "Pachelbel-Live-35bpm.wav")

# ── Systematische Szenarien ────────────────────────────────────────────────────
# N3 – Tempoinvarianz: Noise fixiert auf niedrig, Tempo variiert
SPEED_SCENARIOS  = [0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4]
SPEED_NOISE_FIXED = 0.05

# N2 – Rauschrobustheit: Tempo fixiert auf 1.0, Noise variiert
NOISE_SCENARIOS  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
NOISE_SPEED_FIXED = 1.0


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def frames_to_measures(n_frames: int) -> float:
    """Frame-Index → Takt-Einheiten (Referenz-Partitur-BPM aus settings.py)."""
    sec_per_frame   = HOP_LENGTH / SAMPLE_RATE
    sec_per_measure = (60.0 / BPM) * BEATS_PER_MEASURE
    return (n_frames * sec_per_frame) / sec_per_measure


def stretch_chroma(chroma: np.ndarray, speed_factor: float) -> np.ndarray:
    """Zeit-Stretching: speed_factor > 1 → kürzer, < 1 → länger."""
    return zoom(chroma, (1, 1.0 / speed_factor), order=1)


def prepare_live_chroma(base_chroma: np.ndarray,
                        speed_factor: float,
                        noise_level: float) -> np.ndarray:
    """Simuliertes Live-Chroma: Tempo-Variation + Rauschen + L2-Normalisierung."""
    live = stretch_chroma(base_chroma, speed_factor) if speed_factor != 1.0 \
        else base_chroma.copy()
    if noise_level > 0:
        live = live + np.random.normal(0, noise_level, live.shape)
        live = np.clip(live, 0, None)
    norms = np.linalg.norm(live, axis=0)
    norms[norms == 0] = 1.0
    return live / norms


# ── ODTW-Kern (identisch zu ODTWTracker._odtw_step in dtw.py) ─────────────────

def run_odtw(ref_chroma: np.ndarray, live_chroma: np.ndarray) -> np.ndarray:
    """
    Führt StandardODTW frame-by-frame durch.

    Logik identisch zu ODTWTracker._odtw_step() in dtw.py,
    parameterisiert über settings.py. Recovery und Sleep werden
    hier bewusst ausgelassen, um den ODTW-Kern isoliert zu testen.

    Returns:
        positions: Array der Tracking-Positionen (Frame-Indizes in Referenz).
    """
    n_ref  = ref_chroma.shape[1]
    n_live = live_chroma.shape[1]

    accumulated_costs = np.full(n_ref, np.inf)
    accumulated_costs[0] = 0.0
    current_position = 0

    chroma_buffer = deque(maxlen=max(1, SMOOTHING_WINDOW))
    positions = np.empty(n_live, dtype=int)

    for i in range(n_live):
        # Glättung (Moving Average) + L2-Normalisierung – wie in ODTWTracker
        chroma_buffer.append(live_chroma[:, i])
        avg = np.mean(np.array(chroma_buffer), axis=0)
        norm = np.linalg.norm(avg)
        live_vec = avg / norm if norm > 1e-4 else avg

        # Suchfenster
        start_scan = max(0, current_position - SEARCH_WINDOW)
        end_scan   = min(n_ref, current_position + SEARCH_WINDOW)

        new_costs  = np.full(n_ref, np.inf)
        best_cost  = np.inf
        best_index = current_position

        for j in range(start_scan, end_scan):
            dot  = np.dot(live_vec, ref_chroma[:, j])
            dot  = max(-1.0, min(1.0, dot))
            dist = 1.0 - dot

            cost_wait = accumulated_costs[j]       + WAIT_PENALTY
            cost_step = accumulated_costs[j - 1]   + STEP_PENALTY  if j > 0 else np.inf
            cost_skip = accumulated_costs[j - 2]   + SKIP_PENALTY  if j > 1 else np.inf

            new_costs[j] = dist + min(cost_wait, cost_step, cost_skip) * DAMPING_FACTOR

            if new_costs[j] < best_cost:
                best_cost  = new_costs[j]
                best_index = j

        accumulated_costs = new_costs
        current_position  = best_index
        positions[i]      = best_index

        if i % 500 == 0:
            print(f"\r  {i}/{n_live} Frames …", end="", flush=True)

    print()
    return positions


# ── Metrik-Berechnung ──────────────────────────────────────────────────────────

def compute_metrics(positions: np.ndarray, ref_len: int) -> dict:
    """
    MAE, RMSE und Acc@1T gegen den linearen Idealpfad.

    Idealpfad: Frame i von n_live entspricht Score-Frame i/n_live * ref_len.
    """
    n_live = len(positions)
    ideal_frames = np.linspace(0, ref_len - 1, n_live)

    tracked_m = np.array([frames_to_measures(p) for p in positions])
    ideal_m   = np.array([frames_to_measures(f) for f in ideal_frames])

    error = tracked_m - ideal_m
    mae   = float(np.mean(np.abs(error)))
    rmse  = float(np.sqrt(np.mean(error ** 2)))
    acc1t = float(np.mean(np.abs(error) <= 1.0))

    return dict(
        mae=mae, rmse=rmse, acc1t=acc1t,
        tracked_m=tracked_m,
        ideal_m=ideal_m,
        x_input_m=np.linspace(0, frames_to_measures(ref_len), n_live),
    )


# ── Tabellen-Ausgabe ───────────────────────────────────────────────────────────

def print_table(header: str, param_values, param_label: str, results: list):
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")
    print(f"  {param_label:<12}  {'MAE [T]':>9}  {'RMSE [T]':>9}  {'Acc@1T':>9}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}")
    for val, res in zip(param_values, results):
        print(f"  {val:<12.2f}  {res['mae']:>9.3f}  {res['rmse']:>9.3f}"
              f"  {res['acc1t']:>9.3f}")
    print(f"{'='*60}\n")


# ── Plot-Funktionen ────────────────────────────────────────────────────────────

def _colormap(n: int):
    return [plt.get_cmap('viridis')(i / max(n - 1, 1)) for i in range(n)]


def _plot_panels(ax_path, ax_bar, ax_acc,
                 param_values, results,
                 param_label: str, ref_total_m: float,
                 highlight_idx=None):
    colors = _colormap(len(param_values))

    # Panel 1: Tracking-Pfade
    ax_path.plot([0, ref_total_m], [0, ref_total_m],
                 color='#aab7b8', linewidth=1.6, linestyle='--',
                 label='Idealpfad', zorder=1)
    for idx, (val, res, color) in enumerate(zip(param_values, results, colors)):
        lw    = 2.4 if idx == highlight_idx else 1.2
        alpha = 1.0 if idx == highlight_idx else 0.75
        ax_path.plot(res['x_input_m'], res['tracked_m'],
                     color=color, linewidth=lw, alpha=alpha,
                     label=f'{param_label}={val:.2g}')
    ax_path.set_xlim(0, ref_total_m)
    ax_path.set_ylim(0, ref_total_m * 1.05)
    ax_path.set_xlabel('Eingabe-Position [Takte]', fontsize=11)
    ax_path.set_ylabel('Partitur-Position [Takte]', fontsize=11)
    ax_path.legend(fontsize=7.5, ncol=2, loc='upper left',
                   framealpha=0.8, handlelength=1.5)
    ax_path.grid(True, linewidth=0.4, alpha=0.5)
    ax_path.tick_params(labelsize=9)

    # Panel 2: MAE-Balken + Acc@1T
    x_pos  = np.arange(len(param_values))
    maes   = [r['mae']   for r in results]
    acc1ts = [r['acc1t'] for r in results]
    labels = [f'{v:.2g}' for v in param_values]

    bars = ax_bar.bar(x_pos, maes, color=colors, width=0.6,
                      zorder=2, edgecolor='white', linewidth=0.5)
    if highlight_idx is not None:
        bars[highlight_idx].set_edgecolor('#e74c3c')
        bars[highlight_idx].set_linewidth(2.2)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=8.5, rotation=30, ha='right')
    ax_bar.set_xlabel(param_label, fontsize=11)
    ax_bar.set_ylabel('MAE [Takte]', fontsize=11)
    ax_bar.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax_bar.grid(axis='y', linewidth=0.4, alpha=0.5)
    ax_bar.tick_params(labelsize=9)

    ax_acc.plot(x_pos, acc1ts, color='#e67e22',
                marker='o', markersize=5, linewidth=2.0, zorder=5)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel('Acc@1T', fontsize=11, color='#e67e22')
    ax_acc.tick_params(axis='y', colors='#e67e22', labelsize=9)
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))


def _save_figure(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, name)
    fig.savefig(base + '.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(base + '.png', bbox_inches='tight', dpi=300)
    print(f'Gespeichert: {base}.pdf / .png')
    plt.close(fig)


def plot_tempo_invariance(results, ref_total_m: float):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'N3 – Tempoinvarianz  (σ={SPEED_NOISE_FIXED}, '
        f'SEARCH_WINDOW={SEARCH_WINDOW}, BPM={BPM})',
        fontsize=12, fontweight='bold', y=0.98)
    ax_acc = ax_bot.twinx()
    hi = SPEED_SCENARIOS.index(1.0) if 1.0 in SPEED_SCENARIOS else None
    _plot_panels(ax_top, ax_bot, ax_acc,
                 SPEED_SCENARIOS, results, 'Tempofaktor', ref_total_m, hi)
    ax_top.set_title('Tracking-Pfade bei unterschiedlichem Tempo', fontsize=11, pad=6)
    ax_bot.set_title('Quantitative Metriken pro Tempofaktor', fontsize=11, pad=6)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'robustness_tempo')


def plot_noise_robustness(results, ref_total_m: float):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'N2 – Rauschrobustheit  (Tempo={NOISE_SPEED_FIXED}x, '
        f'SEARCH_WINDOW={SEARCH_WINDOW}, BPM={BPM})',
        fontsize=12, fontweight='bold', y=0.98)
    ax_acc = ax_bot.twinx()
    hi = NOISE_SCENARIOS.index(0.0) if 0.0 in NOISE_SCENARIOS else None
    _plot_panels(ax_top, ax_bot, ax_acc,
                 NOISE_SCENARIOS, results, 'Rauschpegel σ', ref_total_m, hi)
    ax_top.set_title('Tracking-Pfade bei unterschiedlichem Rauschpegel', fontsize=11, pad=6)
    ax_bot.set_title('Quantitative Metriken pro Rauschpegel', fontsize=11, pad=6)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'robustness_noise')


# ── Haupt-Evaluation ───────────────────────────────────────────────────────────

def run_evaluation(ref_chroma: np.ndarray, live_base: np.ndarray,
                   run_tempo: bool, run_noise: bool):
    ref_len     = ref_chroma.shape[1]
    ref_total_m = frames_to_measures(ref_len)

    if run_tempo:
        print(f'\n{"="*60}')
        print(f'  N3 – Tempoinvarianz  (σ={SPEED_NOISE_FIXED} fixiert)')
        print(f'{"="*60}')
        tempo_results = []
        for sf in SPEED_SCENARIOS:
            print(f'\nTempofaktor {sf:.2f}:')
            live = prepare_live_chroma(live_base, sf, SPEED_NOISE_FIXED)
            pos  = run_odtw(ref_chroma, live)
            res  = compute_metrics(pos, ref_len)
            tempo_results.append(res)
            print(f'  MAE={res["mae"]:.3f} T  RMSE={res["rmse"]:.3f} T  '
                  f'Acc@1T={res["acc1t"]:.3f}')
        print_table('N3 – Tempoinvarianz', SPEED_SCENARIOS, 'Tempofaktor', tempo_results)
        plot_tempo_invariance(tempo_results, ref_total_m)

    if run_noise:
        print(f'\n{"="*60}')
        print(f'  N2 – Rauschrobustheit  (Tempo={NOISE_SPEED_FIXED}x fixiert)')
        print(f'{"="*60}')
        noise_results = []
        for nl in NOISE_SCENARIOS:
            print(f'\nRauschpegel σ={nl:.2f}:')
            live = prepare_live_chroma(live_base, NOISE_SPEED_FIXED, nl)
            pos  = run_odtw(ref_chroma, live)
            res  = compute_metrics(pos, ref_len)
            noise_results.append(res)
            print(f'  MAE={res["mae"]:.3f} T  RMSE={res["rmse"]:.3f} T  '
                  f'Acc@1T={res["acc1t"]:.3f}')
        print_table('N2 – Rauschrobustheit', NOISE_SCENARIOS, 'Rauschpegel σ', noise_results)
        plot_noise_robustness(noise_results, ref_total_m)


# ── Chroma-Lader ───────────────────────────────────────────────────────────────

def load_score_chroma(npz_file: str) -> np.ndarray:
    """Lädt Referenz-Chroma aus .npz (wie score_loader.py im Live Page Turner)."""
    print(f'Lade Partitur: {npz_file} …')
    archive    = np.load(npz_file)
    chroma     = archive['chroma']           # (12, N)
    page_ends  = archive['page_end_indices']
    print(f'  Shape: {chroma.shape}, Seitenenden: {page_ends}')
    return chroma


def load_chroma_from_wav(wav_file: str) -> np.ndarray:
    """
    Lädt WAV und extrahiert Chroma (12, N).

    Verwendet exakt dieselben librosa-Parameter wie ChromaExtractor.extract()
    im Live Page Turner (BLOCK_SIZE als n_fft, center=False, tuning=0),
    simuliert aber den Sliding-Window-Effekt des Ring-Buffers via hop_length=HOP_LENGTH.
    """
    print(f'Lade WAV: {wav_file} …')
    y, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=BLOCK_SIZE,
        hop_length=HOP_LENGTH,
        center=False,   # wie ChromaExtractor.extract() in chroma.py
        tuning=0,       # wie ChromaExtractor.extract() in chroma.py
        n_chroma=12,
    )
    return chroma  # (12, N)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Wissenschaftliche Robustheitsevaluation des ODTW (N2 & N3)')
    parser.add_argument('--score', default=SCORE_DATA_PATH,
                        help='Partitur-Datei (.npz)')
    parser.add_argument('--wav', default=LIVE_WAV_FILE,
                        help='Live-Audio WAV-Datei')
    parser.add_argument('--only-tempo', action='store_true',
                        help='Nur N3 (Tempoinvarianz) auswerten')
    parser.add_argument('--only-noise', action='store_true',
                        help='Nur N2 (Rauschrobustheit) auswerten')
    args = parser.parse_args()

    print(f'\n=== ODTW Robustheitstest ===')
    print(f'  settings.py: SEARCH_WINDOW={SEARCH_WINDOW}, DAMPING={DAMPING_FACTOR}, '
          f'WAIT={WAIT_PENALTY}, SKIP={SKIP_PENALTY}, STEP={STEP_PENALTY}, BPM={BPM}')

    try:
        ref_chroma = load_score_chroma(args.score)
    except Exception as exc:
        print(f'Fehler beim Laden der Partitur: {exc}')
        return

    try:
        live_base = load_chroma_from_wav(args.wav)
    except FileNotFoundError:
        print(f'WAV-Datei nicht gefunden: {args.wav}')
        return
    except Exception as exc:
        print(f'Fehler beim Laden der WAV: {exc}')
        return

    print(f'  Referenz: {ref_chroma.shape[1]} Frames '
          f'({frames_to_measures(ref_chroma.shape[1]):.1f} Takte)')
    print(f'  Live-Basis: {live_base.shape[1]} Frames')

    run_evaluation(ref_chroma, live_base,
                   run_tempo=not args.only_noise,
                   run_noise=not args.only_tempo)
    print('\nFertig.')


if __name__ == '__main__':
    main()
