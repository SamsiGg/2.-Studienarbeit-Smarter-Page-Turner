# =============================================================================
# dtw_illustration.py – Illustration des klassischen (Offline-)DTW
# =============================================================================
# Ausgabe: dtw_illustration.pdf + dtw_illustration.png
#
# Achsen-Konvention (konsistent mit sdtw_illustration.py und odtw_illustration.py):
#   x-Achse = j  (Sequenz X, Länge N)  → Sequenz X unten
#   y-Achse = i  (Sequenz Y, Länge M)  → Sequenz Y links
#   D[i, j]: i = Zeile = y-Achse, j = Spalte = x-Achse  ✓
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(7)

# ── Parameter ─────────────────────────────────────────────────────────────────
N = 10   # Länge Sequenz X  (x-Achse, j-Index)
M = 10   # Länge Sequenz Y  (y-Achse, i-Index)

OUT_BASE = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/dtw_illustration"

# ── Synthetische Sequenzen ────────────────────────────────────────────────────
# Beide Sequenzen haben ähnliche Struktur (zwei Peaks, ein Tal), aber
# unterschiedliches Timing → macht das Warping anschaulich notwendig
x_signal = np.array([0.15, 0.70, 1.00, 0.55, 0.20, 0.35, 0.80, 0.95, 0.50, 0.10])
y_signal = np.array([0.10, 0.25, 0.55, 0.95, 1.00, 0.40, 0.15, 0.60, 0.90, 0.45])

# ── Lokale Kostenmatrix ───────────────────────────────────────────────────────
# Gewünschter Pfad: nicht-linear, klare horizontale und vertikale Segmente
# D[i, j]: i = Zeile = y (Sequenz Y), j = Spalte = x (Sequenz X)
# Gültige DTW-Schritte: (i+1,j+1) diagonal | (i,j+1) j-Schritt (rechts) | (i+1,j) i-Schritt (hoch)
# Im Plot (x=j, y=i): j-Schritt = rechts | i-Schritt = hoch | diagonal = schräg
desired_path = [
    (1, 1), (1, 2), (1, 3),      # rechts: X läuft vor, Y wartet
    (2, 4),                       # diagonal
    (3, 4), (4, 4),               # hoch: Y holt auf, X wartet
    (5, 5),                       # diagonal
    (5, 6), (5, 7),               # rechts
    (6, 8),                       # diagonal
    (7, 8), (8, 8),               # hoch
    (9, 9),                       # diagonal
    (9, 10), (10, 10),            # rechts, dann hoch
]

# D hat Größe (M+1) × (N+1); D[i, j]: i = Zeile = y (Sequenz Y), j = Spalte = x (Sequenz X)
local_cost = 0.80 + np.random.rand(M, N) * 0.15

for (i, j) in desired_path:
    local_cost[i - 1, j - 1] = 0.02 + np.random.rand() * 0.05
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i - 1 + di, j - 1 + dj
        if 0 <= ni < M and 0 <= nj < N:
            local_cost[ni, nj] = min(local_cost[ni, nj], 0.18 + np.random.rand() * 0.08)

# ── Klassisches DTW: Akkumulierte Kostenmatrix ────────────────────────────────
D = np.full((M + 1, N + 1), np.inf)
D[0, 0] = 0.0

for i in range(1, M + 1):
    for j in range(1, N + 1):
        prev = min(D[i-1, j-1], D[i-1, j], D[i, j-1])
        D[i, j] = local_cost[i-1, j-1] + prev

# ── Optimaler Pfad (Traceback) ────────────────────────────────────────────────
def traceback(D, max_i, max_j):
    path, i, j = [], max_i, max_j
    while i >= 1 and j >= 1:
        path.append((i, j))
        if i == 1 and j == 1:
            break
        cands = [(ii, jj) for ii, jj in [(i-1, j-1), (i-1, j), (i, j-1)]
                 if ii >= 1 and jj >= 1]
        if not cands:
            break
        i, j = min(cands, key=lambda x: D[x[0], x[1]])
    return list(reversed(path))

opt_path = traceback(D, M, N)

# ── Kostenmatrix für Plot ─────────────────────────────────────────────────────
# D_norm[i-1, j-1] = Kosten an (i,j)
# x-Achse = j, y-Achse = i  → kein Transponieren nötig
# origin='lower': Zeile 0 = unten = i=1  ✓
D_cost = D[1:, 1:].copy()
D_cost[D_cost == np.inf] = np.nanmax(D_cost[D_cost != np.inf])
D_norm = (D_cost - D_cost.min()) / (D_cost.max() - D_cost.min())
D_display = D_norm   # shape (M, N): rows=i (y), cols=j (x)

cmap_cost = LinearSegmentedColormap.from_list(
    "cost", ["#eaf4fb", "#aed6f1", "#2e86c1", "#1a5276", "#0b1f36"]
)

# ── FIGUR ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 9))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    2, 3,
    width_ratios=[1.4, 5, 0.25],
    height_ratios=[5, 1.4],
    hspace=0.04, wspace=0.04,
    left=0.06, right=0.96, top=0.96, bottom=0.06
)

ax_main   = fig.add_subplot(gs[0, 1])
ax_left   = fig.add_subplot(gs[0, 0], sharey=ax_main)   # Sequenz Y (vertikal)
ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)   # Sequenz X (horizontal)
ax_cbar   = fig.add_subplot(gs[0, 2])

# ── Hauptheatmap ──────────────────────────────────────────────────────────────
# extent: x = j (1..N), y = i (1..M)
im = ax_main.imshow(D_display, origin='lower', aspect='auto',
                    cmap=cmap_cost, vmin=0, vmax=1,
                    extent=[0.5, N + 0.5, 0.5, M + 0.5])

for x in np.arange(0.5, N + 1.5, 1):
    ax_main.axvline(x, color='white', linewidth=0.4, zorder=2)
for y in np.arange(0.5, M + 1.5, 1):
    ax_main.axhline(y, color='white', linewidth=0.4, zorder=2)

# ── Lineare Referenz (Diagonale) ──────────────────────────────────────────────
ax_main.plot([1, N], [1, M], color='#aab7b8', linewidth=1.8,
             linestyle='--', zorder=4, alpha=0.8)
ax_main.text(N * 0.35, M * 0.35 - 0.5, 'lineare Ausrichtung',
             color='#7f8c8d', fontsize=9, ha='center', va='top',
             rotation=45, clip_on=True)

# ── Optimaler Warpingpfad ─────────────────────────────────────────────────────
orange = '#e67e22'
xs_opt = [j for (i, j) in opt_path]   # x-Achse = j (Sequenz X)
ys_opt = [i for (i, j) in opt_path]   # y-Achse = i (Sequenz Y)

ax_main.plot(xs_opt, ys_opt, color=orange, linewidth=3.5,
             linestyle='-', zorder=6, solid_capstyle='round', solid_joinstyle='round')

# Start (1,1)
ax_main.plot(1, 1, 's', color='#27ae60', markersize=11, zorder=7)
ax_main.text(1.3, 1.4, '$(1,\,1)$\nfester Start',
             color='#27ae60', fontsize=9, fontweight='bold', ha='left', va='bottom', zorder=8)

# Ende (N,M)
ax_main.plot(N, M, 's', color='#c0392b', markersize=11, zorder=7)
ax_main.text(N - 0.3, M - 0.4, '$(N,\,M)$\nfester Endpunkt',
             color='#c0392b', fontsize=9, fontweight='bold', ha='right', va='top', zorder=8)

# Pfad-Label
mid = len(opt_path) // 2
ax_main.text(xs_opt[mid] + 0.2, ys_opt[mid] + 0.65,
             'optimaler Warpingpfad',
             color=orange, fontsize=10, fontweight='bold',
             ha='center', va='bottom', zorder=8)

ax_main.set_xlim(0.5, N + 0.5)
ax_main.set_ylim(0.5, M + 0.5)
ax_main.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax_main.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax_main.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax_main.tick_params(axis='both', labelsize=9)
ax_main.set_xlabel(r'Sequenz $X$ (Frame $j$)', fontsize=11, labelpad=4)
ax_main.set_ylabel(r'Sequenz $Y$ (Frame $i$)', fontsize=11, labelpad=4)

# ── Linkes Panel: Sequenz Y (vertikal, i auf y-Achse) ────────────────────────
i_coords = np.arange(1, M + 1)
ax_left.plot(y_signal, i_coords, color='#2e86c1', linewidth=2.0)
ax_left.fill_betweenx(i_coords, 0, y_signal, alpha=0.15, color='#2e86c1')
ax_left.set_xlim(0, 1.4)
ax_left.invert_xaxis()              # 0 → rechts (Matrixseite), Amplitude wächst links
ax_left.set_ylim(0.5, M + 0.5)
for sp in ['top', 'right', 'bottom']:
    ax_left.spines[sp].set_visible(False)
ax_left.spines['left'].set_color('#888')
ax_left.set_xticks([])
ax_left.set_yticks([])
ax_left.set_xlabel('Amplitude', fontsize=8, color='#666', labelpad=3)
ax_left.set_ylabel('Sequenz $Y$', fontsize=11, color='#2e86c1',
                   fontweight='bold', labelpad=8)

# ── Unteres Panel: Sequenz X (horizontal, j auf x-Achse) ─────────────────────
j_coords = np.arange(1, N + 1)
ax_bottom.plot(j_coords, x_signal, color='#2e86c1', linewidth=2.0)
ax_bottom.fill_between(j_coords, 0, x_signal, alpha=0.15, color='#2e86c1')
ax_bottom.set_xlim(0.5, N + 0.5)
ax_bottom.set_ylim(0, 1.4)
ax_bottom.invert_yaxis()            # 0 → oben (Matrixseite), Amplitude wächst unten
for sp in ['top', 'right', 'left']:
    ax_bottom.spines[sp].set_visible(False)
ax_bottom.spines['bottom'].set_color('#888')
ax_bottom.set_yticks([])
ax_bottom.set_xticks([])
ax_bottom.set_ylabel('Amplitude', fontsize=8, color='#666', labelpad=3)
ax_bottom.set_xlabel('Sequenz $X$', fontsize=11, color='#2e86c1',
                     fontweight='bold', labelpad=8)

# ── Colorbar ───────────────────────────────────────────────────────────────────
fig.colorbar(im, cax=ax_cbar)
ax_cbar.set_ylabel('akkumulierte Kosten (normiert)', fontsize=9, labelpad=6)
ax_cbar.tick_params(labelsize=8)

# ── Speichern ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
fig.savefig(OUT_BASE + ".pdf", bbox_inches='tight', dpi=300)
fig.savefig(OUT_BASE + ".png", bbox_inches='tight', dpi=300)
print(f"Gespeichert:\n  {OUT_BASE}.pdf\n  {OUT_BASE}.png")
plt.show()
