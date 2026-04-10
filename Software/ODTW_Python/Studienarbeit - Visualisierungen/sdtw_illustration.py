# =============================================================================
# sdtw_illustration.py – Illustration des Subsequence DTW
# =============================================================================
# Ausgabe: sdtw_illustration.pdf + sdtw_illustration.png
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(12)

# ── Parameter ────────────────────────────────────────────────────────────────
J = 16   # Länge Referenz X  (breite Achse)
I = 6    # Länge Suchsequenz Y (kurze Achse)

OUT_BASE = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/sdtw_illustration"

# ── Lokale Kostenmatrix (synthetisch) ────────────────────────────────────────
local_cost = 0.75 + np.random.rand(I, J) * 0.2

# Günstiger Korridor (optimaler Pfad soll diagonal durch j=8..13 laufen)
for i_idx in range(I):
    j_center = 8 + i_idx
    for dj in [-1, 0, 1]:
        jc = j_center + dj
        if 0 <= jc < J:
            local_cost[i_idx, jc] = 0.04 + np.random.rand() * 0.08

# ── sDTW: Akkumulierte Kostenmatrix ──────────────────────────────────────────
# D hat Größe (I+1) × (J+1); Index [i, j] entspricht Zelle D(i,j)
D = np.full((I + 1, J + 1), np.inf)
D[0, :] = 0.0          # D(0, j) = 0  →  freier Einstieg
# D[:, 0] = inf  →  bereits durch np.full gesetzt

for i in range(1, I + 1):
    for j in range(1, J + 1):
        prev = min(D[i-1, j-1], D[i-1, j], D[i, j-1])
        D[i, j] = local_cost[i-1, j-1] + prev

# ── Optimaler Pfad (Traceback) ───────────────────────────────────────────────
j_end = int(np.argmin(D[I, 1:])) + 1

def traceback(D, j_end, I):
    path, i, j = [], I, j_end
    while i >= 1:
        path.append((i, j))
        if i == 1:
            break
        cands = [(ii, jj) for ii, jj in [(i-1, j-1), (i-1, j), (i, j-1)]
                 if ii >= 1 and jj >= 1]
        i, j = min(cands, key=lambda x: D[x[0], x[1]])
    return list(reversed(path))

opt_path = traceback(D, j_end, I)
j_start = opt_path[0][1]

# ── Akkumulierte Kostenmatrix normieren (nur i=1..I, j=1..J) ─────────────────
D_cost = D[1:, 1:].copy()    # shape (I, J); Zeile 0 = i=1 (unten im Plot)
D_cost[D_cost == np.inf] = np.nanmax(D_cost[D_cost != np.inf])
D_norm = (D_cost - D_cost.min()) / (D_cost.max() - D_cost.min())
# origin='lower' → array-Zeile 0 = Plotunten = i=1  ✓ (kein Flip nötig)

cmap_cost = LinearSegmentedColormap.from_list(
    "cost", ["#eaf4fb", "#aed6f1", "#2e86c1", "#1a5276", "#0b1f36"]
)

# ── FIGUR ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.8))
fig.patch.set_facecolor('white')
plt.subplots_adjust(left=0.10, right=0.86, top=0.82, bottom=0.13)

# ── Hauptheatmap: akkumulierte Kosten (i=1..I, j=1..J) ──────────────────────
# Koordinatensystem: x = j (0..J), y = i (0..I), Zelle (i,j) → Mittelpunkt (j, i)
im = ax.imshow(D_norm, origin='lower', aspect='auto',
               cmap=cmap_cost, vmin=0, vmax=1,
               extent=[0.5, J + 0.5, 0.5, I + 0.5])

# ── Zellenraster ─────────────────────────────────────────────────────────────
for x in np.arange(0.5, J + 1.5, 1):
    ax.axvline(x, color='white', linewidth=0.4, zorder=2)
for y in np.arange(0.5, I + 1.5, 1):
    ax.axhline(y, color='white', linewidth=0.4, zorder=2)

# ── Zeile i=0: D(0,j) = 0 – freier Einstieg (grün) ──────────────────────────
green = '#27ae60'
ax.add_patch(mpatches.Rectangle(
    (0.5, -0.5), J, 1.0,
    facecolor='#d5f5e3', edgecolor=green, linewidth=1.2, zorder=3, clip_on=False
))
# Vertikale Trennlinien innerhalb der Zeile
for x in np.arange(0.5, J + 1.5, 1):
    ax.plot([x, x], [-0.5, 0.5], color=green, linewidth=0.4, zorder=4, clip_on=False)
ax.text((J + 1) / 2, 0.0,
        r'$D(0,\,j) = 0\quad$ (freier Einstieg an beliebiger Stelle)',
        color=green, fontsize=9.5, fontweight='bold',
        ha='center', va='center', zorder=5, clip_on=False)

# ── Spalte j=0: D(i,0) = ∞ (rot) ────────────────────────────────────────────
red = '#c0392b'
ax.add_patch(mpatches.Rectangle(
    (-0.5, -0.5), 1.0, I + 1.0,
    facecolor='#fadbd8', edgecolor=red, linewidth=1.2, zorder=3, clip_on=False
))
for y in np.arange(-0.5, I + 1.5, 1):
    ax.plot([-0.5, 0.5], [y, y], color=red, linewidth=0.4, zorder=4, clip_on=False)
ax.text(0.0, (I + 1) / 2,
        r'$D(i,\,0) = \infty$',
        color=red, fontsize=9.5, fontweight='bold',
        ha='center', va='center', rotation=90, zorder=5, clip_on=False)


# ── Optimaler Pfad ───────────────────────────────────────────────────────────
orange = '#e67e22'
xs_opt = [j for (i, j) in opt_path]
ys_opt = [i for (i, j) in opt_path]

ax.plot(xs_opt, ys_opt,
        color=orange, linewidth=3.5, linestyle='-', zorder=6,
        solid_capstyle='round', solid_joinstyle='round')
ax.plot(xs_opt[0],  ys_opt[0],  'o', color='white', markersize=9,
        markeredgecolor=orange, markeredgewidth=2.2, zorder=7)
ax.plot(xs_opt[-1], ys_opt[-1], 'o', color=orange, markersize=9, zorder=7)

# Label in Pfadmitte
mid = len(opt_path) // 2
ax.text(xs_opt[mid] - 0.1, ys_opt[mid] + 0.55,
        'optimaler Pfad',
        color=orange, fontsize=9.5, fontweight='bold',
        ha='center', va='bottom', clip_on=False)

# ── Referenzbalken oben: zeigt wo die Subsequenz gefunden wurde ──────────────
bar_y      = I + 0.65
bar_height = 0.55

# Gesamte Referenz X (grau)
ax.add_patch(mpatches.Rectangle(
    (0.5, bar_y), J, bar_height,
    facecolor='#eaecee', edgecolor='#aab7b8', linewidth=0.9,
    zorder=3, clip_on=False
))
# Gematchter Abschnitt (orange)
ax.add_patch(mpatches.Rectangle(
    (j_start - 0.5, bar_y), xs_opt[-1] - j_start + 1, bar_height,
    facecolor='#f5cba7', edgecolor=orange, linewidth=1.3,
    zorder=4, clip_on=False
))
# Label: gefundene Subsequenz (innerhalb des orangefarbenen Balkens)
ax.text((j_start - 0.5 + xs_opt[-1] + 0.5) / 2, bar_y + bar_height / 2,
        'gefundene Subsequenz',
        color='#a04000', fontsize=8.5, fontweight='bold',
        ha='center', va='center', clip_on=False, zorder=6)

# ── Achsen ───────────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, J + 0.5)
ax.set_ylim(-0.5, I + 0.5)

ax.set_xlabel(r'Referenz $X$ (Frame $j$)', fontsize=11, labelpad=6)
ax.set_ylabel(r'Suchsequenz $Y$ (Frame $i$)', fontsize=11, labelpad=6)
# Kein Titel

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', labelsize=9)

# Colorbar (nur für den Hauptbereich)
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.01)
cbar.set_label('akkumulierte Kosten (normiert)', fontsize=9.5)
cbar.ax.tick_params(labelsize=9)

# ── Speichern ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
fig.savefig(OUT_BASE + ".pdf", bbox_inches='tight', dpi=300)
fig.savefig(OUT_BASE + ".png", bbox_inches='tight', dpi=300)
print(f"Gespeichert:\n  {OUT_BASE}.pdf\n  {OUT_BASE}.png")
plt.show()
