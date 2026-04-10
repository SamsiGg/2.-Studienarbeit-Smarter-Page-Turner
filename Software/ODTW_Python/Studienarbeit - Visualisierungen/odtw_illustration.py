# =============================================================================
# odtw_illustration.py – Illustration des Online DTW (ODTW)
# =============================================================================
# Ausgabe: odtw_illustration.pdf + odtw_illustration.png
#
# Achsen-Konvention (nach Dixon 2005):
#   x-Achse = j  (Referenzpartitur, vollständig bekannt, links → rechts)
#   y-Achse = i  (Live-Audio-Frame, wächst inkrementell nach oben)
#
# Drei Zeilen von unten nach oben:
#   i-1  (unten)  – Vergangenheit, bereits berechnet
#   i    (Mitte)  – Gegenwart, wird gerade berechnet
#   i+1  (oben)   – Zukunft, Audio existiert noch nicht
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

OUT_BASE = "/Users/samuelgeffert/Documents/Programmieren/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/generated/odtw_illustration"

# ── Layout-Parameter ──────────────────────────────────────────────────────────
N_J         = 11   # Anzahl Partitur-Positionen (x-Achse, j = 1..N_J)
CELL_W      = 1.0  # Zellbreite  (x, pro Partiturposition j)
CELL_H      = 1.5  # Zellhöhe   (y, pro Live-Zeile)

corridor_lo = 3    # Suchkorridor linke Grenze j
corridor_hi = 9    # Suchkorridor rechte Grenze j
arrow_j     = 5    # Zielspalte j für Wait/Step/Skip-Pfeile
min_j       = 8    # Minimum-Spalte j (geschätzte Partiturposition)

ROW_PAST = 0   # Zeile i-1 (unten)
ROW_CURR = 1   # Zeile i   (Mitte)
ROW_FUT  = 2   # Zeile i+1 (oben)

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def col_center(j):
    return (j - 1) * CELL_W + CELL_W / 2

def row_center(row_idx):
    return row_idx * CELL_H + CELL_H / 2

def cell_bl(j, row_idx):
    return (j - 1) * CELL_W, row_idx * CELL_H

# ── Figur ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(-2.8, 13.5)
ax.set_ylim(-2.8, 6.5)

def draw_cell(j, row_idx, fc, ec='#85929e', lw=0.8, ls='-', alpha=1.0, zorder=2):
    x, y = cell_bl(j, row_idx)
    ax.add_patch(Rectangle((x, y), CELL_W, CELL_H,
                            facecolor=fc, edgecolor=ec, linewidth=lw,
                            linestyle=ls, alpha=alpha, zorder=zorder))

# ── Zellen zeichnen ───────────────────────────────────────────────────────────
for j in range(1, N_J + 1):
    # Zeile i-1: berechnet → grau
    draw_cell(j, ROW_PAST, '#d5d8dc', ec='#7f8c8d')

    # Zeile i: Korridor hellblau, außerhalb sehr hell
    if corridor_lo <= j <= corridor_hi:
        draw_cell(j, ROW_CURR, '#eaf4fb', ec='#7f8c8d')
    else:
        draw_cell(j, ROW_CURR, '#fbfcfc', ec='#bdc3c7', lw=0.5)

    # Zeile i+1: Zukunft → gestrichelt, transparent
    draw_cell(j, ROW_FUT, '#f8f8f8', ec='#bdc3c7', lw=0.6, ls='--', alpha=0.55)

# ── Suchkorridor-Rahmen ───────────────────────────────────────────────────────
x_cl, y_cl = cell_bl(corridor_lo, ROW_CURR)
w_corr = (corridor_hi - corridor_lo + 1) * CELL_W
ax.add_patch(Rectangle((x_cl, y_cl), w_corr, CELL_H,
                        facecolor='none', edgecolor='#2e86c1', linewidth=2.8, zorder=5))

ax.annotate('Suchkorridor',
            xy=(x_cl + w_corr, row_center(ROW_CURR)),
            xytext=(x_cl + w_corr + 0.4, row_center(ROW_CURR)),
            fontsize=9.5, color='#2e86c1', fontweight='bold', va='center',
            arrowprops=dict(arrowstyle='->', color='#2e86c1', lw=1.6))

# ── Ziel-Zelle D(i,j) ────────────────────────────────────────────────────────
draw_cell(arrow_j, ROW_CURR, '#fdebd0', ec='#e67e22', lw=2.2, zorder=6)
ax.text(col_center(arrow_j), row_center(ROW_CURR), r'$D(i,j)$',
        color='#e67e22', fontsize=9, fontweight='bold',
        ha='center', va='center', zorder=10)

# ── Pfeile: Wait / Step / Skip ────────────────────────────────────────────────
COLORS = {'Wait': '#8e44ad', 'Step': '#27ae60', 'Skip': '#c0392b'}
OFFSET = 0.38   # Pfeil-Endpunkte etwas vom Zellmittelpunkt entfernt

def draw_option(src_j, label):
    color = COLORS[label]
    sx, sy = col_center(src_j), row_center(ROW_PAST)
    dx, dy = col_center(arrow_j), row_center(ROW_CURR)

    # Quell-Zelle farbig hinterlegen
    draw_cell(src_j, ROW_PAST, facecolor_for(color), ec=color, lw=2.0, zorder=4)

    # Label in der Quell-Zelle
    ax.text(sx, sy, label, color=color, fontsize=9.5, fontweight='bold',
            ha='center', va='center', zorder=9)

    # Pfeil von Quell-Zelle zur Ziel-Zelle
    ax.annotate('',
                xy=(dx, dy - OFFSET),
                xytext=(sx, sy + OFFSET),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.2),
                zorder=8)

def facecolor_for(color):
    return {'#8e44ad': '#f5eef8', '#27ae60': '#eafaf1', '#c0392b': '#fdedec'}[color]

draw_option(arrow_j,     'Wait')   # (i-1, j)   → vertikal
draw_option(arrow_j - 1, 'Step')   # (i-1, j-1) → diagonal
draw_option(arrow_j - 2, 'Skip')   # (i-1, j-2) → steiler diagonal

# ── Minimum-Zelle ─────────────────────────────────────────────────────────────
draw_cell(min_j, ROW_CURR, '#d5f5e3', ec='#1e8449', lw=2.8, zorder=6)
mx = col_center(min_j)
my = row_center(ROW_CURR)
ax.plot(mx, my, '*', color='#1e8449', markersize=16, zorder=10)


# Textbox oberhalb der Minimum-Zelle auf Höhe von Zeile i+1
text_y = ROW_FUT * CELL_H + CELL_H / 2   # Mittelpunkt von Zeile i+1
ax.text(mx, text_y,
        'Minimum der Zeile $i$\n= geschätzte Partiturposition',
        color='#1e8449', fontsize=9, fontweight='bold',
        ha='center', va='center', zorder=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#eafaf1',
                  edgecolor='#1e8449', alpha=0.92))

# ── Zeilen-Labels (linke Seite) ───────────────────────────────────────────────
lx = -0.15
for row_idx, lbl, sub, color in [
    (ROW_PAST, '$i-1$', 'berechnet', '#555555'),
    (ROW_CURR, '$i$',   'aktuell',   '#2c3e50'),
    (ROW_FUT,  '$i+1$', 'Zukunft',   '#aaaaaa'),
]:
    cy = row_center(row_idx)
    ax.text(lx, cy + 0.22, lbl,
            ha='right', va='center', fontsize=13, fontweight='bold', color=color)
    ax.text(lx, cy - 0.28, sub,
            ha='right', va='center', fontsize=8.5, color=color, style='italic')

# ── Achsen ────────────────────────────────────────────────────────────────────
# x-Achse an y=0 (untere Gitterkante)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

x_ticks = [col_center(j) for j in range(1, N_J + 1)]
ax.set_xticks(x_ticks)
ax.set_xticklabels([str(j) for j in range(1, N_J + 1)], fontsize=9)
ax.tick_params(axis='x', direction='out', pad=4)

ax.set_yticks([])
ax.set_xlabel('Referenzpartitur $X$ (Position $j$)', fontsize=12, labelpad=8)
ax.set_ylabel('Live-Audio-Frame $Y$ (Position $i$)', fontsize=12, labelpad=8)


# ── Speichern ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
fig.savefig(OUT_BASE + ".pdf", bbox_inches='tight', dpi=300)
fig.savefig(OUT_BASE + ".png", bbox_inches='tight', dpi=300)
print(f"Gespeichert:\n  {OUT_BASE}.pdf\n  {OUT_BASE}.png")
plt.show()
