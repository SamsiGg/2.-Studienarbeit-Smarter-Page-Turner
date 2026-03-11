# =============================================================================
# dtw.py – ODTW-Tracker mit Seitenwechsel-Erkennung und optionalem Recovery
# =============================================================================
# Teensy-Äquivalent: DTW.h (DTWTracker Klasse)
#   - Teensy: Zwei-Spalten-Ansatz mit Pointer-Swap + Kosten-Normalisierung
#   - Python: Akkumulierte Kosten mit Damping Factor
#   Beide Varianten erzeugen gleichwertiges Tracking-Verhalten.
#
# Kombiniert:
#   - StandardODTW.step() Logik (aus dtw_engine.py)
#   - Page-Turn-Erkennung (aus DTW.h checkPageTurn())
#   - Start-Schwellen-Erkennung (aus DTW.h update())
#   - Optionales Recovery via Moving Average + Full-Score-Scan
# =============================================================================

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

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
    print("[ODTWTracker] Numba verfügbar – JIT-kompilierter DTW-Scan aktiv.")

except ImportError:
    _NUMBA_AVAILABLE = False
    print("[ODTWTracker] Numba nicht gefunden – numpy-Fallback aktiv.")


@dataclass
class TrackerState:
    """Snapshot des Tracker-Zustands für die GUI.

    Wird vom Worker-Thread erzeugt und per Queue an die GUI übergeben.
    """
    current_position: int       # Aktueller Frame-Index in der Partitur
    current_page: int           # Aktuelle Seite (1-basiert)
    total_pages: int            # Gesamtanzahl Seiten
    score_length: int           # Gesamtframes in der Partitur
    progress_fraction: float    # 0.0 bis 1.0
    current_cost: float         # Akkumulierte Kosten an aktueller Position
    rms_level: float            # Audio-Energielevel
    is_running: bool            # Ob Tracking gestartet hat
    is_finished: bool           # Ob das Stück zu Ende ist
    page_turn_triggered: bool   # True wenn gerade umgeblättert wurde
    page_turn_target: int       # Zielseite (wenn umgeblättert)
    measure: int                # Geschätzter Takt
    beat: int                   # Geschätzter Schlag im Takt
    recovered: bool             # True wenn in diesem Schritt Recovery stattfand
    is_sleeping: bool = False   # True wenn Stille erkannt → Tracker pausiert
    live_chroma: Optional[np.ndarray] = None  # (12,) L2-normalisierter Live-Chroma-Vektor
    ref_chroma: Optional[np.ndarray] = None   # (12,) Referenz-Chroma an aktueller Position


class ODTWTracker:
    """Online Dynamic Time Warping Tracker mit Seitenwechsel-Erkennung.

    Teensy-Äquivalent: DTWTracker in DTW.h

    Args:
        reference_chroma: Shape (12, N) numpy Array.
        page_end_indices: Liste der Frame-Indizes an Seitenenden.
        Alle weiteren Parameter aus settings.py.
    """

    def __init__(
        self,
        reference_chroma: np.ndarray,
        page_end_indices: list[int],
        search_window: int,
        damping_factor: float,
        wait_penalty: float,
        skip_penalty: float,
        step_penalty: float,
        start_threshold_rms: float,
        smoothing_window: int,
        bpm: int,
        beats_per_measure: int,
        hop_length: int,
        sample_rate: int,
        use_recovery: bool = False,
        recovery_threshold: float = 10.5,
        recovery_avg_window: int = 300,
        recovery_buffer_size: int = 500,
        recovery_jump_threshold: float = 0.4,
        silence_frames_before_sleep: int = 50,
    ):
        # Referenz
        self.ref = reference_chroma
        self.n_frames = reference_chroma.shape[1]

        # Seitenwechsel
        self.page_end_indices = page_end_indices
        self.total_pages = len(page_end_indices) + 1

        # ODTW-Parameter
        self.search_window = search_window
        self.damping = damping_factor
        self.penalty_wait = wait_penalty
        self.penalty_skip = skip_penalty
        self.penalty_step = step_penalty

        # Start-Erkennung
        self.start_threshold_rms = start_threshold_rms

        # Glättung
        self.chroma_buffer = deque(maxlen=max(1, smoothing_window))

        # Musikalisch
        self.bpm = bpm
        self.beats_per_measure = beats_per_measure
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Recovery
        self.use_recovery = use_recovery
        self.recovery_threshold = recovery_threshold
        self.recovery_jump_threshold = recovery_jump_threshold
        self.cost_history = deque(maxlen=recovery_avg_window)
        self.chroma_history = deque(maxlen=recovery_buffer_size)
        self.recovery_count = 0
        # Sleep-Modus
        self.silence_frames_before_sleep = silence_frames_before_sleep

        # Zustand
        self._last_live_chroma: Optional[np.ndarray] = None
        self._reset_state()

    def _reset_state(self):
        """Internen Zustand zurücksetzen."""
        self.current_position = 0
        self.accumulated_costs = np.full(self.n_frames, np.inf)
        self.accumulated_costs[0] = 0.0
        self._last_page = 1
        self.is_running = False
        self.is_finished = False
        self.is_sleeping = False
        self._silent_frame_count = 0
        self.chroma_buffer.clear()
        self.cost_history.clear()
        self.chroma_history.clear()
        self.recovery_count = 0

    def reset(self):
        """Tracker auf Anfangszustand zurücksetzen."""
        self._reset_state()

    def step(self, live_chroma: np.ndarray, rms_level: float) -> TrackerState:
        """Einen Chroma-Frame verarbeiten und aktualisierten Zustand zurückgeben.

        Teensy-Äquivalent: DTWTracker::update() + checkPageTurn()

        Args:
            live_chroma: Roher Chroma-Vektor, Shape (12,).
            rms_level: Aktuelle RMS-Audioenergie.

        Returns:
            TrackerState mit allen GUI-relevanten Informationen.
        """
        page_turn_triggered = False
        page_turn_target = 0
        recovered = False

        playing = rms_level >= self.start_threshold_rms

        # 1. Start-Erkennung (erster Start oder Aufwachen aus Sleep)
        if not self.is_running:
            if playing:
                self.is_running = True
                self._silent_frame_count = 0
                print(f"[ODTW] Tracking gestartet (RMS={rms_level:.4f})")

        elif self.is_sleeping:
            # Im Sleep-Modus: warten auf Musik
            if playing:
                # Aufwachen: von Position 0 neu starten
                # → Recovery springt automatisch zur richtigen Stelle wenn nötig
                self._reset_state()
                self.is_running = True
                print(f"[ODTW] Aufgewacht – starte von Position 0")
            # Kein ODTW während Sleep

        else:
            # Aktives Tracking: Stille-Zähler pflegen
            if playing:
                self._silent_frame_count = 0
            else:
                self._silent_frame_count += 1
                if self._silent_frame_count >= self.silence_frames_before_sleep:
                    self.is_sleeping = True
                    print(f"[ODTW] Sleep-Modus (Stille seit "
                          f"{self._silent_frame_count} Frames)")

        # 2. ODTW-Schritt (nur wenn aktiv und nicht schlafend)
        current_cost = 0.0
        if self.is_running and not self.is_sleeping:
            current_cost, recovered = self._odtw_step(live_chroma)

        # 3. Aktuelle Seite bestimmen
        current_page = self._get_current_page()

        # 4. Seitenwechsel: Seite hat sich durch normales ODTW geändert (nicht Recovery)
        if self.is_running and not self.is_sleeping and not recovered:
            if current_page != self._last_page:
                page_turn_triggered = True
                page_turn_target = current_page
                print(f"\n>>> BLÄTTERN zu Seite {current_page} <<<\n")
        self._last_page = current_page

        # 5. Takt/Schlag berechnen
        measure, beat = self._calculate_measure_beat()

        # 6. Fortschritt
        progress = self.current_position / max(1, self.n_frames)

        ref_ch = self.ref[:, self.current_position].copy() if self.is_running else None

        return TrackerState(
            current_position=self.current_position,
            current_page=current_page,
            total_pages=self.total_pages,
            score_length=self.n_frames,
            progress_fraction=min(1.0, progress),
            current_cost=current_cost,
            rms_level=rms_level,
            is_running=self.is_running,
            is_finished=self.is_finished,
            page_turn_triggered=page_turn_triggered,
            page_turn_target=page_turn_target,
            measure=measure,
            beat=beat,
            recovered=recovered,
            is_sleeping=self.is_sleeping,
            live_chroma=self._last_live_chroma,
            ref_chroma=ref_ch,
        )

    def _odtw_step(self, live_chroma_raw: np.ndarray) -> tuple[float, bool]:
        """Kern-ODTW-Logik mit optionalem Recovery.

        Returns:
            (best_cost, recovered)
        """
        recovered = False

        # Chroma für Recovery-Scan speichern (vor Glättung)
        if self.use_recovery:
            self.chroma_history.append(live_chroma_raw.copy())

        # Glättung (Moving Average)
        self.chroma_buffer.append(live_chroma_raw)
        avg_chroma = np.mean(np.array(self.chroma_buffer), axis=0)

        # L2-Normalisierung
        norm = np.linalg.norm(avg_chroma)
        if norm > 1e-4:
            live_vec = avg_chroma / norm
        else:
            live_vec = avg_chroma
        self._last_live_chroma = live_vec

        # Suchfenster
        start_scan = max(0, self.current_position - self.search_window)
        end_scan = min(self.n_frames, self.current_position + self.search_window)

        new_costs = np.full(self.n_frames, np.inf)
        best_cost = np.inf
        best_index = self.current_position

        for i in range(start_scan, end_scan):
            dot_prod = np.dot(live_vec, self.ref[:, i])
            dot_prod = max(-1.0, min(1.0, dot_prod))
            local_dist = 1.0 - dot_prod

            cost_wait = self.accumulated_costs[i] + self.penalty_wait
            cost_step = np.inf
            if i > 0:
                cost_step = self.accumulated_costs[i - 1] + self.penalty_step
            cost_skip = np.inf
            if i > 1:
                cost_skip = self.accumulated_costs[i - 2] + self.penalty_skip

            min_prev = min(cost_wait, cost_step, cost_skip)
            new_costs[i] = local_dist + (min_prev * self.damping)

            if new_costs[i] < best_cost:
                best_cost = new_costs[i]
                best_index = i

        self.accumulated_costs = new_costs
        self.current_position = best_index

        # Recovery-Check
        if self.use_recovery:
            self.cost_history.append(best_cost)

            if len(self.cost_history) == self.cost_history.maxlen:
                avg_cost = np.mean(self.cost_history)

                if avg_cost > self.recovery_threshold:
                    t_scan = time.time()
                    new_pos, scan_quality = self._full_score_scan()
                    scan_ms = (time.time() - t_scan) * 1000
                    self.cost_history.clear()

                    if scan_quality < self.recovery_jump_threshold:
                        self._reset_at_position(new_pos)
                        self.recovery_count += 1
                        recovered = True

                        # Nochmal step an neuer Position für saubere Kosten
                        best_cost = self._single_step(live_vec, new_pos)
                        print(f"\n  [RECOVERY #{self.recovery_count}] "
                              f"Sprung zu Frame {new_pos} "
                              f"(Avg-Kosten > {self.recovery_threshold:.1f}, "
                              f"Qualität: {scan_quality:.2f}) "
                              f"| Scan: {scan_ms:.0f} ms")
                    else:
                        print(f"\n  [RECOVERY] Kein gültiger Ort – warte auf Musik "
                              f"(Qualität: {scan_quality:.2f} > "
                              f"{self.recovery_jump_threshold:.2f}) "
                              f"| Scan: {scan_ms:.0f} ms")

        return best_cost, recovered

    def _single_step(self, live_vec: np.ndarray, position: int) -> float:
        """Einzelner ODTW-Step nach einem Reset – gibt nur Kosten zurück."""
        start_scan = max(0, position - self.search_window)
        end_scan = min(self.n_frames, position + self.search_window)

        new_costs = np.full(self.n_frames, np.inf)
        best_cost = np.inf
        best_index = position

        for i in range(start_scan, end_scan):
            dot_prod = np.dot(live_vec, self.ref[:, i])
            dot_prod = max(-1.0, min(1.0, dot_prod))
            local_dist = 1.0 - dot_prod

            cost_wait = self.accumulated_costs[i] + self.penalty_wait
            cost_step = np.inf
            if i > 0:
                cost_step = self.accumulated_costs[i - 1] + self.penalty_step
            cost_skip = np.inf
            if i > 1:
                cost_skip = self.accumulated_costs[i - 2] + self.penalty_skip

            min_prev = min(cost_wait, cost_step, cost_skip)
            new_costs[i] = local_dist + (min_prev * self.damping)

            if new_costs[i] < best_cost:
                best_cost = new_costs[i]
                best_index = i

        self.accumulated_costs = new_costs
        self.current_position = best_index
        return best_cost

    def _full_score_scan(self) -> tuple[int, float]:
        """Scannt den gesamten Score via Subsequence-DTW gegen die letzten N Live-Frames.

        Toleriert Tempoabweichungen durch elastische Ausrichtung (DTW-Pfade:
        diagonal, vertikal, horizontal). Siehe recovery_odtw.py für den
        ausführlichen Studienarbeit-Hinweis zu dieser Methode.

        Returns:
            (beste Endposition, Qualitätswert): Qualität = mittlere Kosinus-Distanz
            an bester Position (0.0 = perfekter Match, 1.0 = kein Match / Stille).
        """
        history = list(self.chroma_history)
        n_history = len(history)

        if n_history == 0:
            return self.current_position, 1.0

        live_normalized = []
        for frame in history:
            norm = np.linalg.norm(frame)
            live_normalized.append(frame / norm if norm > 1e-4 else frame)
        live_matrix = np.array(live_normalized)  # (N, 12)

        n = n_history

        # Referenz auf jeden 10. Frame reduzieren → ~10x schneller.
        # ODTW korrigiert die genaue Position nach dem Reset von selbst.
        step = 10
        ref_sub = self.ref[:, ::step]  # (12, m_sub)
        m_sub = ref_sub.shape[1]

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

        # Qualität: mittlere Kosinus-Distanz der Live-Frames zur besten Position
        # 0.0 = perfekter Match, 1.0 = kein Match (Stille/Rauschen)
        scan_quality = float(D[:, best_sub].mean())

        return min(best_sub * step, self.n_frames - 1), scan_quality

    def _reset_at_position(self, new_position: int):
        """Setzt den Tracker an einer neuen Position neu."""
        new_position = max(0, min(new_position, self.n_frames - 1))
        self.accumulated_costs = np.full(self.n_frames, np.inf)
        self.accumulated_costs[new_position] = 0.0
        self.current_position = new_position
        self.chroma_buffer.clear()
        # _last_page auf neue Position setzen, damit kein falscher Blätterbefehl
        # im selben Schritt ausgelöst wird (recovered=True blockiert ihn ohnehin,
        # aber nach dem Reset stimmt der Referenzwert nun)
        self._last_page = self._get_current_page()

    def _get_current_page(self) -> int:
        """Aktuelle Seite basierend auf Position bestimmen (1-basiert)."""
        for i, end_idx in enumerate(self.page_end_indices):
            if self.current_position < end_idx:
                return i + 1
        return self.total_pages

    def _calculate_measure_beat(self) -> tuple[int, int]:
        """Frame-Index in Takt und Schlag umrechnen."""
        time_sec = (self.current_position * self.hop_length) / self.sample_rate
        sec_per_beat = 60.0 / self.bpm
        sec_per_measure = sec_per_beat * self.beats_per_measure

        measure = int(time_sec / sec_per_measure) + 1
        beat = int((time_sec % sec_per_measure) / sec_per_beat) + 1

        return measure, beat
