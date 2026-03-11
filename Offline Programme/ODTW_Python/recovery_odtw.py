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

import time
import numpy as np
from collections import deque
from dtw_engine import StandardODTW

# --- RECOVERY-PARAMETER ---
RECOVERY_THRESHOLD = 38        # Moving-Average-Kosten ab denen Recovery triggert
RECOVERY_AVG_WINDOW = 300      # Fenstergröße für Moving Average (wie im Plot)
RECOVERY_BUFFER_SIZE = 300     # Anzahl gespeicherter Chroma-Frames für Full-Scan
SCAN_STEP = 10                 # Subsampling-Faktor für Full-Score-Scan (~10x schneller)

# --- NUMBA-JIT (optional, deutliche Beschleunigung der inneren DTW-Schleifen) ---
try:
    from numba import njit

    @njit
    def _dtw_inner(D):
        """Subsequence-DTW-Kern, Numba-JIT-kompiliert.

        Drei Pfade: diagonal (live+ref vorrücken), vertikal (nur live vorrückt),
        horizontal (nur ref vorrückt). Freier Start: dp[0, :] = 0.

        Hinweis: Erste Ausführung triggert JIT-Kompilierung (~1–3 s einmalig).

        Args:
            D: Kosinus-Distanzmatrix, Shape (n, m_sub), dtype float64.

        Returns:
            Bester Endindex in der (subsampled) Referenz.
        """
        n, m_sub = D.shape
        dp = np.empty((n + 1, m_sub))
        for ii in range(n + 1):
            for jj in range(m_sub):
                dp[ii, jj] = np.inf
        for jj in range(m_sub):
            dp[0, jj] = 0.0

        for i in range(1, n + 1):
            # Vertikal: j=0 (kein Diagonal möglich)
            dp[i, 0] = D[i - 1, 0] + dp[i - 1, 0]
            # Diagonal + Vertikal: min(dp[i-1,j], dp[i-1,j-1]) + d
            for j in range(1, m_sub):
                prev = dp[i - 1, j] if dp[i - 1, j] < dp[i - 1, j - 1] else dp[i - 1, j - 1]
                dp[i, j] = D[i - 1, j] + prev
            # Horizontal: links-nach-rechts Propagation
            for j in range(1, m_sub):
                c = dp[i, j - 1] + D[i - 1, j]
                if c < dp[i, j]:
                    dp[i, j] = c

        # argmin der letzten Zeile
        best = 0
        best_val = dp[n, 0]
        for j in range(1, m_sub):
            if dp[n, j] < best_val:
                best_val = dp[n, j]
                best = j
        return best

    # JIT-Kompilierung vorab triggern, damit der erste echte Recovery-Scan
    # nicht durch die Compile-Zeit (~100 ms) gebremst wird.
    _dummy = np.zeros((2, 2), dtype=np.float64)
    _dtw_inner(_dummy)
    del _dummy

    _NUMBA_AVAILABLE = True
    print("[RecoveryODTW] Numba verfügbar – JIT-kompilierter DTW-Scan aktiv.")

except ImportError:
    _NUMBA_AVAILABLE = False
    print("[RecoveryODTW] Numba nicht gefunden – numpy-Fallback aktiv.")


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
                t_scan = time.time()
                new_pos = self._full_score_scan()
                scan_ms = (time.time() - t_scan) * 1000

                self._reset_at_position(new_pos)
                self.cost_history.clear()
                self.recovery_count += 1
                recovered = True

                # Nochmal step an neuer Position für saubere Kosten
                index, cost = self.odtw.step(live_chroma_raw)
                print(f"\n  [RECOVERY #{self.recovery_count}] "
                      f"Sprung zu Frame {new_pos} (Avg-Kosten > {self.recovery_threshold}) "
                      f"| Scan: {scan_ms:.0f} ms")

        return index, cost, recovered

    def _full_score_scan(self):
        """Scannt den gesamten Score via Subsequence-DTW gegen die letzten N Live-Frames.

        # -------------------------------------------------------------------------
        # STUDIENARBEIT-HINWEIS:
        # Der ursprüngliche lineare Scan verglich N Live-Frames 1:1 mit N
        # Referenz-Frames. Das versagt bei Tempoabweichungen, weil z.B. bei 1.3x
        # Tempo dieselbe Melodie auf weniger Frames verteilt ist und das Fenster
        # deshalb systematisch an die falsche Stelle zeigt.
        #
        # Die Lösung: Subsequence-DTW mit freiem Start. Anstelle eines festen
        # Fensters lässt das DTW zu, dass live[i] und ref[j] elastisch aufeinander
        # abgebildet werden (drei Pfade: diagonal, vertikal, horizontal). Dadurch
        # wird Tempovarianz beim Recovery-Scan toleriert – analog zur Elastizität
        # des ODTW-Algorithmus selbst.
        #
        # Komplexität: O(N × m) mit N = History-Länge, m = Referenzlänge.
        # Recovery triggert selten, daher ist diese Rechenzeit akzeptabel.
        # -------------------------------------------------------------------------

        Returns:
            Beste Endposition in der Referenz (Frame-Index).
        """
        history = list(self.chroma_history)
        n_history = len(history)

        if n_history == 0:
            return self.odtw.current_position_index

        # Live-Frames normalisieren
        live_normalized = []
        for frame in history:
            norm = np.linalg.norm(frame)
            live_normalized.append(frame / norm if norm > 0.0001 else frame)
        live_matrix = np.array(live_normalized)  # (N, 12)

        n = n_history

        # Referenz auf jeden SCAN_STEP-ten Frame reduzieren → ~SCAN_STEP-fache Beschleunigung.
        # Die genaue Position spielt keine Rolle – ODTW fängt sich nach dem Reset von selbst.
        ref_sub = self.ref[:, ::SCAN_STEP]  # (12, m_sub)
        m_sub = ref_sub.shape[1]

        # Kosinus-Distanzmatrix: (N, m_sub) via Matrizenmultiplikation (effizient)
        D = (1.0 - np.clip(live_matrix @ ref_sub, -1.0, 1.0)).astype(np.float64)  # (N, m_sub)

        # Subsequence-DTW mit freiem Start (numba-JIT wenn verfügbar):
        if _NUMBA_AVAILABLE:
            best_sub = int(_dtw_inner(D))
        else:
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
            best_sub = int(np.argmin(dp[n, :]))

        # Beste Endposition zurück auf originalen Frame-Index mappen
        return min(best_sub * SCAN_STEP, self.n_frames_ref - 1)

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
