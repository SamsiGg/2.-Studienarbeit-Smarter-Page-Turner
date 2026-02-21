# =============================================================================
# recovery_odtw.py – ODTW mit automatischer Recovery bei verlorenem Tracking
# =============================================================================
# Wrapped StandardODTW und überwacht die globalen Kosten per Moving Average.
# Wenn der Durchschnitt einen Threshold überschreitet, wird ein Full-Score-Scan
# durchgeführt, um die wahrscheinlichste Position zu finden.
#
# Nutzung:
#   from recovery_odtw import RecoveryODTW
#   engine = RecoveryODTW(reference_chroma)
#   index, cost, recovered = engine.step(live_chroma_raw)
# =============================================================================

import numpy as np
from collections import deque
from dtw_engine import StandardODTW

# --- RECOVERY-PARAMETER ---
RECOVERY_THRESHOLD = 10.5        # Moving-Average-Kosten ab denen Recovery triggert
RECOVERY_AVG_WINDOW = 300      # Fenstergröße für Moving Average (wie im Plot)
RECOVERY_BUFFER_SIZE = 500     # Anzahl gespeicherter Chroma-Frames für Full-Scan


class RecoveryODTW:
    """ODTW mit automatischer Positionswiederherstellung.

    Nutzt intern StandardODTW für normales Tracking. Überwacht die globalen
    Kosten per Moving Average. Wenn der Durchschnitt den Threshold überschreitet,
    wird ein Full-Score-Scan ausgeführt und der Algorithmus neu gestartet.
    """

    def __init__(self, reference_chroma,
                 recovery_threshold=RECOVERY_THRESHOLD,
                 avg_window=RECOVERY_AVG_WINDOW,
                 buffer_size=RECOVERY_BUFFER_SIZE):
        self.ref = reference_chroma
        self.n_frames_ref = reference_chroma.shape[1]

        # Interner ODTW-Engine
        self.odtw = StandardODTW(reference_chroma)

        # Recovery-Einstellungen
        self.recovery_threshold = recovery_threshold

        # Moving Average für Kosten
        self.cost_history = deque(maxlen=avg_window)

        # Chroma-History für Full-Score-Scan
        self.chroma_history = deque(maxlen=buffer_size)

        # Stats
        self.recovery_count = 0

    def step(self, live_chroma_raw):
        """Ein Tracking-Schritt mit Recovery-Überwachung.

        Returns:
            (index, cost, recovered):
                index: Aktuelle Position im Score.
                cost: Globale Kosten.
                recovered: True wenn in diesem Schritt Recovery stattfand.
        """
        # Live-Chroma für mögliche Recovery speichern
        self.chroma_history.append(live_chroma_raw.copy())

        # Normaler ODTW-Step
        index, cost = self.odtw.step(live_chroma_raw)

        # Kosten im Moving-Average-Buffer speichern
        self.cost_history.append(cost)

        # Recovery-Check: Erst wenn genug Daten für den Moving Average da sind
        recovered = False
        if len(self.cost_history) == self.cost_history.maxlen:
            avg_cost = np.mean(self.cost_history)

            if avg_cost > self.recovery_threshold:
                new_pos = self._full_score_scan()
                self._reset_at_position(new_pos)
                self.cost_history.clear()
                self.recovery_count += 1
                recovered = True

                # Nochmal step an neuer Position für saubere Kosten
                index, cost = self.odtw.step(live_chroma_raw)
                print(f"\n  [RECOVERY #{self.recovery_count}] "
                      f"Sprung zu Frame {new_pos} (Avg-Kosten > {self.recovery_threshold})")

        return index, cost, recovered

    def _full_score_scan(self):
        """Scannt den gesamten Score gegen die letzten N Live-Frames.

        Berechnet für jede Startposition im Score die durchschnittliche
        Cosinus-Distanz über die gespeicherten Frames (Sliding Window).

        Returns:
            Beste Startposition (Frame-Index).
        """
        history = list(self.chroma_history)
        n_history = len(history)

        if n_history == 0:
            return self.odtw.current_position_index

        # Live-Frames normalisieren
        live_normalized = []
        for frame in history:
            norm = np.linalg.norm(frame)
            if norm > 0.0001:
                live_normalized.append(frame / norm)
            else:
                live_normalized.append(frame)
        live_matrix = np.array(live_normalized)  # (N, 12)

        # Für jede mögliche Startposition: Durchschnittliche Distanz
        n_ref = self.n_frames_ref
        best_pos = 0
        best_avg_dist = np.inf

        max_start = n_ref - n_history
        if max_start < 0:
            max_start = 0

        for start in range(max_start + 1):
            # Referenz-Slice: (12, N) → transponiert zu (N, 12)
            ref_slice = self.ref[:, start:start + n_history].T  # (N, 12)

            # Kosinus-Distanz pro Frame: 1 - dot(live, ref)
            dots = np.sum(live_matrix * ref_slice, axis=1)
            np.clip(dots, -1.0, 1.0, out=dots)
            distances = 1.0 - dots  # (N,)

            avg_dist = np.mean(distances)

            if avg_dist < best_avg_dist:
                best_avg_dist = avg_dist
                best_pos = start

        # Position am Ende des Fensters zurückgeben (= wo wir jetzt sind)
        return best_pos + n_history - 1

    def _reset_at_position(self, new_position):
        """Setzt den ODTW-Engine an einer neuen Position zurück."""
        new_position = max(0, min(new_position, self.n_frames_ref - 1))

        self.odtw.accumulated_costs = np.full(self.n_frames_ref, np.inf)
        self.odtw.accumulated_costs[new_position] = 0
        self.odtw.current_position_index = new_position
        self.odtw.chroma_buffer.clear()

    @property
    def current_position_index(self):
        return self.odtw.current_position_index
