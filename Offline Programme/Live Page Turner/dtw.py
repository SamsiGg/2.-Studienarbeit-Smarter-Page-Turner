# =============================================================================
# dtw.py – ODTW-Tracker mit Seitenwechsel-Erkennung
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
# =============================================================================

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


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
        page_turn_offset: int,
        start_threshold_rms: float,
        smoothing_window: int,
        bpm: int,
        beats_per_measure: int,
        hop_length: int,
        sample_rate: int,
    ):
        # Referenz
        self.ref = reference_chroma
        self.n_frames = reference_chroma.shape[1]

        # Seitenwechsel
        self.page_end_indices = page_end_indices
        self.num_turns = len(page_end_indices)       # Anzahl Umblätter-Punkte
        self.total_pages = len(page_end_indices) + 1  # 3 Grenzen → 4 Seiten
        self.page_turn_offset = page_turn_offset

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

        # Zustand
        self._reset_state()

    def _reset_state(self):
        """Internen Zustand zurücksetzen."""
        self.current_position = 0
        self.accumulated_costs = np.full(self.n_frames, np.inf)
        self.accumulated_costs[0] = 0.0
        self.next_page_idx = 0       # Index in page_end_indices
        self.is_running = False
        self.is_finished = False
        self.chroma_buffer.clear()

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

        # 1. Start-Erkennung (Teensy: volume > START_THRESHOLD)
        if not self.is_running:
            if rms_level >= self.start_threshold_rms:
                self.is_running = True
                print(f"[ODTW] Tracking gestartet (RMS={rms_level:.4f})")

        # 2. ODTW-Schritt (nur wenn aktiv)
        current_cost = 0.0
        if self.is_running and not self.is_finished:
            self.current_position, current_cost = self._odtw_step(live_chroma)

            # 3. Seitenwechsel prüfen
            new_page = self._check_page_turn()
            if new_page is not None:
                page_turn_triggered = True
                page_turn_target = new_page

        # 4. Takt/Schlag berechnen
        measure, beat = self._calculate_measure_beat()

        # 5. Aktuelle Seite bestimmen
        current_page = self._get_current_page()

        # 6. Fortschritt
        progress = self.current_position / max(1, self.n_frames)

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
        )

    def _odtw_step(self, live_chroma_raw: np.ndarray) -> tuple[int, float]:
        """Kern-ODTW-Logik. Spiegelt StandardODTW.step() aus dtw_engine.py.

        Returns:
            (best_index, best_cost)
        """
        # Glättung (Moving Average)
        self.chroma_buffer.append(live_chroma_raw)
        avg_chroma = np.mean(np.array(self.chroma_buffer), axis=0)

        # L2-Normalisierung
        norm = np.linalg.norm(avg_chroma)
        if norm > 1e-4:
            live_vec = avg_chroma / norm
        else:
            live_vec = avg_chroma

        # Suchfenster
        start_scan = max(0, self.current_position - self.search_window)
        end_scan = min(self.n_frames, self.current_position + self.search_window)

        new_costs = np.full(self.n_frames, np.inf)
        best_cost = np.inf
        best_index = self.current_position

        for i in range(start_scan, end_scan):
            # A: Lokale Distanz (Kosinus-Distanz: 0.0 bis 2.0)
            dot_prod = np.dot(live_vec, self.ref[:, i])
            dot_prod = max(-1.0, min(1.0, dot_prod))
            local_dist = 1.0 - dot_prod

            # B: Pfad-Auswahl (Minimum der Vorgänger + Penalty)
            cost_wait = self.accumulated_costs[i] + self.penalty_wait
            cost_step = np.inf
            if i > 0:
                cost_step = self.accumulated_costs[i - 1] + self.penalty_step
            cost_skip = np.inf
            if i > 1:
                cost_skip = self.accumulated_costs[i - 2] + self.penalty_skip

            min_prev = min(cost_wait, cost_step, cost_skip)

            # C: Gesamtkosten mit Damping
            new_costs[i] = local_dist + (min_prev * self.damping)

            # D: Besten Index finden
            if new_costs[i] < best_cost:
                best_cost = new_costs[i]
                best_index = i

        # Zustand updaten
        self.accumulated_costs = new_costs
        self.current_position = best_index

        return best_index, best_cost

    def _check_page_turn(self) -> Optional[int]:
        """Prüft ob ein Seitenwechsel ausgelöst werden soll.

        Teensy-Äquivalent: DTWTracker::checkPageTurn() in DTW.h

        Returns:
            Neue Seitennummer (1-basiert) wenn umgeblättert, sonst None.
        """
        if self.next_page_idx >= self.num_turns:
            if not self.is_finished and self.current_position >= self.n_frames - 5:
                self.is_finished = True
                print("[ODTW] Stück beendet.")
            return None

        threshold = self.page_end_indices[self.next_page_idx] - self.page_turn_offset

        if self.current_position >= threshold:
            self.next_page_idx += 1
            new_page = self.next_page_idx + 1  # 1-basiert: nach 1. Umblättern → Seite 2
            print(f"\n>>> BLÄTTERN zu Seite {new_page} <<<\n")
            return new_page

        return None

    def _get_current_page(self) -> int:
        """Aktuelle Seite basierend auf Position bestimmen (1-basiert)."""
        for i, end_idx in enumerate(self.page_end_indices):
            if self.current_position < end_idx:
                return i + 1
        return self.total_pages

    def _calculate_measure_beat(self) -> tuple[int, int]:
        """Frame-Index in Takt und Schlag umrechnen.

        Wiederverwendet Logik aus dtw_engine.py calculate_current_measure().
        """
        time_sec = (self.current_position * self.hop_length) / self.sample_rate
        sec_per_beat = 60.0 / self.bpm
        sec_per_measure = sec_per_beat * self.beats_per_measure

        measure = int(time_sec / sec_per_measure) + 1
        beat = int((time_sec % sec_per_measure) / sec_per_beat) + 1

        return measure, beat
