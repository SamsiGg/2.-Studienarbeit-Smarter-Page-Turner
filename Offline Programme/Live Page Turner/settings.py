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
DAMPING_FACTOR = 0.99      # Dämpfung alter Kosten (Bayesianisch optimiert)
WAIT_PENALTY = 0.1        # Strafe fürs Stehenbleiben (Bayesianisch optimiert)
SKIP_PENALTY = 0.03        # Strafe fürs Überspringen  (Bayesianisch optimiert)
STEP_PENALTY = 0.0           # Diagonaler Schritt kostet nichts

# --- Recovery ---
USE_RECOVERY = True             # True: RecoveryODTW, False: Standard-ODTW
RECOVERY_THRESHOLD = 38        # Moving-Average-Kosten ab denen Recovery triggert
RECOVERY_AVG_WINDOW = 300        # Fenstergröße Moving Average (Frames)
RECOVERY_BUFFER_SIZE = 300       # Gespeicherte Chroma-Frames für Full-Score-Scan
RECOVERY_JUMP_THRESHOLD = 1   # Max. Scan-Qualität für gültigen Sprung (0=perfekt, 1=Stille)

# --- Seitenwechsel ---
PAGE_TURN_OFFSET = 100        # Frames VOR page_end_index auslösen
PAGE_TURN_STABLE_TIME = 0.0  # Sekunden ohne Recovery, bevor MIDI-Signal gesendet wird

# --- Start-Erkennung & Sleep-Modus ---
START_THRESHOLD_RMS = 0.01   # RMS-Schwelle zum Starten (Teensy: START_THRESHOLD = 1500, int16-Skala)
SILENCE_FRAMES_BEFORE_SLEEP = 500  # Aufeinanderfolgende leise Frames → Sleep (~0.6s bei 512/44100)

# --- Musikalischer Kontext ---
BPM = 35                     # Tempo des Stücks
BEATS_PER_MEASURE = 4        # Taktart (4/4)

# --- Glättung ---
SMOOTHING_WINDOW = 1         # Moving Average für Chroma (1 = aus)

# --- Dateipfade ---
from pathlib import Path
SCORE_DATA_PATH = str(Path(__file__).parent.parent / "data" / "scores" / "Pachelbel_Musescore.npz")

# --- Bluetooth MIDI (Mac → iPad) ---
USE_BLE_KEYBOARD = True      # True: Seitenumblättern via Bluetooth MIDI
                             # Voraussetzung: pip install mido python-rtmidi
                             # Einrichtung: Audio MIDI Setup → Bluetooth → iPad verbinden

# --- GUI ---
GUI_UPDATE_MS = 50           # GUI-Poll-Intervall in ms (20 Hz)
PAGE_TURN_FLASH_MS = 2000    # Dauer der Blätter-Benachrichtigung in ms
RECOVERY_FLASH_MS = 1000     # Dauer der Recovery-Benachrichtigung in ms
SHOW_NOTATION = False        # True: Noten via verovio anzeigen (benötigt verovio + cairosvg)
