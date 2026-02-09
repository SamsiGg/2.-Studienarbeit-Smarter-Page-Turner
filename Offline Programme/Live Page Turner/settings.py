# =============================================================================
# settings.py – Konfiguration für den Live Page Turner
# =============================================================================
# Alle Parameter an einer Stelle. Spiegelt die Teensy Settings.h.
#
# Teensy-Äquivalente in Kommentaren:
#   Settings.h:  FFT_SIZE, PENALTY_WAIT/STEP/SKIP, CALC_RADIUS, etc.
#   Unterschied: Teensy nutzt Kosten-Normalisierung (min subtrahieren),
#                Python nutzt Damping Factor (multiplikative Dämpfung).
#                Daher unterschiedliche Penalty-Skalen.
# =============================================================================

# --- Audio ---
SAMPLE_RATE = 44100          # Hz
BLOCK_SIZE = 4096            # FFT-Fenstergröße  (Teensy: FFT_SIZE = 4096)
HOP_LENGTH = 512             # Ring-Buffer Hop    (Teensy: kein Overlap, verarbeitet vollen Block)
NUM_CHROMA = 12              # Tonklassen C..B    (Teensy: NUM_CHROMA = 12)

# --- ODTW-Algorithmus ---
SEARCH_WINDOW = 100          # Suchradius ±Frames (Teensy: CALC_RADIUS = 100)
DAMPING_FACTOR = 0.96        # Dämpfung alter Kosten (Teensy: nutzt Normalisierung statt Damping)
WAIT_PENALTY = 0.4           # Strafe fürs Stehenbleiben (Teensy: PENALTY_WAIT = 2.0)
SKIP_PENALTY = 0.2           # Strafe fürs Überspringen  (Teensy: PENALTY_SKIP = 0.8)
STEP_PENALTY = 0.0           # Diagonaler Schritt kostet nichts (Teensy: PENALTY_STEP = 0.0)

# Optimierte Werte (Bayesianische Optimierung, 108 Trials):
#   DAMPING_FACTOR = 0.8977
#   WAIT_PENALTY   = 0.9366
#   SKIP_PENALTY   = 0.1898
#   SEARCH_WINDOW  = 100

# --- Seitenwechsel ---
PAGE_TURN_OFFSET = 10        # Frames VOR page_end_index auslösen (Teensy: PAGE_TURN_OFFSET = 10)

# --- Start-Erkennung ---
START_THRESHOLD_RMS = 0.01   # RMS-Schwelle zum Starten (Teensy: START_THRESHOLD = 1500, int16-Skala)

# --- Musikalischer Kontext ---
BPM = 40                     # Tempo des Stücks
BEATS_PER_MEASURE = 4        # Taktart (4/4)

# --- Glättung ---
SMOOTHING_WINDOW = 1         # Moving Average für Chroma (1 = aus)

# --- Dateipfade ---
SCORE_DATA_PATH = "../ODTW_Python/data/ScoreData.h"

# --- GUI ---
GUI_UPDATE_MS = 50           # GUI-Poll-Intervall in ms (20 Hz)
PAGE_TURN_FLASH_MS = 2000    # Dauer der Blätter-Benachrichtigung in ms
