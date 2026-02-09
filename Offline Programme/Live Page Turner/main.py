# =============================================================================
# main.py – Live Page Turner: tkinter GUI + Audio-Verarbeitung
# =============================================================================
# Entry-Point des Programms.
#
# Teensy-Äquivalent: odtw_turner.cpp (setup + loop)
#   - Dort: I2S Mikrofon → Buffer → Chroma → DTW → Serial ('n'/'p')
#   - Hier:  sounddevice  → Buffer → Chroma → DTW → GUI + Terminal
#
# Threading-Modell:
#   Main Thread:   tkinter GUI (root.mainloop)
#   Worker Thread: Mikrofon → RingBuffer → Chroma → ODTW → Queue
#   Kommunikation: queue.Queue(maxsize=5), threading.Event für Stop
# =============================================================================

import sys
import threading
import queue
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd

import settings
from score_loader import load_score_data
from chroma import AudioRingBuffer, ChromaExtractor
from dtw import ODTWTracker, TrackerState


# =============================================================================
# Audio-Worker-Thread
# =============================================================================

class AudioProcessingThread(threading.Thread):
    """Worker-Thread: Mikrofon → RingBuffer → Chroma → ODTW → State-Queue.

    Spiegelt die loop() Funktion in odtw_turner.cpp.
    """

    def __init__(self, tracker: ODTWTracker, state_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.state_queue = state_queue
        self.stop_event = stop_event

    def run(self):
        ring_buffer = AudioRingBuffer(settings.BLOCK_SIZE)
        extractor = ChromaExtractor(settings.SAMPLE_RATE, settings.BLOCK_SIZE)

        try:
            with sd.InputStream(
                channels=1,
                samplerate=settings.SAMPLE_RATE,
                blocksize=settings.HOP_LENGTH,
                dtype='float32',
            ) as stream:
                print("[Audio] Stream gestartet.")

                while not self.stop_event.is_set():
                    # 1. Audio lesen (512 Samples ~ 11ms)
                    data, overflowed = stream.read(settings.HOP_LENGTH)
                    if overflowed:
                        print("[Audio] WARNUNG: Buffer Overflow!")
                    mono = data[:, 0]

                    # 2. In Ringpuffer schieben
                    ring_buffer.append(mono)

                    # 3. Chroma aus vollem Buffer berechnen
                    audio_buf = ring_buffer.get()
                    rms = extractor.compute_rms(audio_buf)
                    chroma_vec = extractor.extract(audio_buf)

                    # 4. ODTW-Schritt
                    state = self.tracker.step(chroma_vec, rms)

                    # 5. State in Queue (älteste verwerfen wenn voll)
                    try:
                        self.state_queue.put_nowait(state)
                    except queue.Full:
                        try:
                            self.state_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.state_queue.put_nowait(state)

                    # Stück fertig?
                    if state.is_finished:
                        print("[Audio] Stück beendet, Thread stoppt.")
                        break

        except Exception as e:
            print(f"[Audio] FEHLER: {e}")

        print("[Audio] Thread beendet.")


# =============================================================================
# tkinter GUI
# =============================================================================

class PageTurnerGUI:
    """Hauptfenster des Live Page Turners."""

    def __init__(self, root: tk.Tk, score_data):
        self.root = root
        self.score_data = score_data

        # Thread-Kommunikation
        self.state_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.worker_thread: AudioProcessingThread | None = None

        # Flash-Timer
        self._flash_after_id = None

        self._build_ui()

    def _build_ui(self):
        self.root.title("Live Page Turner")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        # Farben
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        green = "#a6e3a1"
        surface = "#313244"

        main_frame = tk.Frame(self.root, bg=bg, padx=20, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Status ---
        status_frame = tk.Frame(main_frame, bg=surface, padx=10, pady=6)
        status_frame.pack(fill=tk.X, pady=(0, 12))

        self.status_dot = tk.Label(status_frame, text="\u25cf", font=("Helvetica", 14),
                                   fg="#6c7086", bg=surface)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))

        self.status_label = tk.Label(status_frame, text="Bereit", font=("Helvetica", 13),
                                     fg=fg, bg=surface)
        self.status_label.pack(side=tk.LEFT)

        # --- Seitenanzeige ---
        page_frame = tk.Frame(main_frame, bg=bg, pady=10)
        page_frame.pack(fill=tk.X)

        tk.Label(page_frame, text="Seite", font=("Helvetica", 16), fg="#6c7086",
                 bg=bg).pack()

        self.page_label = tk.Label(page_frame, text=f"1 / {self.score_data.num_pages}",
                                   font=("Helvetica", 52, "bold"), fg=fg, bg=bg)
        self.page_label.pack()

        # --- Fortschrittsbalken ---
        progress_frame = tk.Frame(main_frame, bg=bg, pady=8)
        progress_frame.pack(fill=tk.X)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=surface, background=accent,
                        thickness=18, borderwidth=0)

        self.progress_bar = ttk.Progressbar(progress_frame, style="Custom.Horizontal.TProgressbar",
                                            maximum=100, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, padx=5)

        self.progress_label = tk.Label(progress_frame, text="0.0 %",
                                       font=("Helvetica", 12), fg="#6c7086", bg=bg)
        self.progress_label.pack(pady=(4, 0))

        # --- Takt/Schlag ---
        self.measure_label = tk.Label(main_frame, text="Takt 1, Schlag 1",
                                      font=("Helvetica", 15), fg=fg, bg=bg)
        self.measure_label.pack(pady=(8, 8))

        # --- Audio Level ---
        level_frame = tk.Frame(main_frame, bg=bg, pady=4)
        level_frame.pack(fill=tk.X)

        tk.Label(level_frame, text="Audio", font=("Helvetica", 11), fg="#6c7086",
                 bg=bg).pack(side=tk.LEFT, padx=(5, 8))

        self.level_canvas = tk.Canvas(level_frame, width=360, height=16,
                                      bg=surface, highlightthickness=0)
        self.level_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, 16, fill=green, width=0)

        # --- Buttons ---
        btn_frame = tk.Frame(main_frame, bg=bg, pady=12)
        btn_frame.pack(fill=tk.X)

        btn_style = {"font": ("Helvetica", 14, "bold"), "width": 12, "pady": 6,
                     "relief": tk.FLAT, "cursor": "hand2"}

        self.start_btn = tk.Button(btn_frame, text="START", bg=green, fg="#1e1e2e",
                                   activebackground="#94e2d5", command=self._on_start,
                                   **btn_style)
        self.start_btn.pack(side=tk.LEFT, expand=True, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="STOP", bg="#f38ba8", fg="#1e1e2e",
                                  activebackground="#eba0ac", command=self._on_stop,
                                  state=tk.DISABLED, **btn_style)
        self.stop_btn.pack(side=tk.LEFT, expand=True, padx=5)

        # --- Page-Turn-Notification ---
        self.notification_frame = tk.Frame(main_frame, bg=bg, pady=8)
        self.notification_frame.pack(fill=tk.X)

        self.notification_label = tk.Label(self.notification_frame, text="",
                                           font=("Helvetica", 16, "bold"),
                                           fg="#1e1e2e", bg=bg, pady=8)
        self.notification_label.pack(fill=tk.X)

        # --- Cost-Anzeige (klein, unten) ---
        self.cost_label = tk.Label(main_frame, text="Cost: --",
                                   font=("Helvetica", 10), fg="#585b70", bg=bg)
        self.cost_label.pack(pady=(4, 0))

    def _on_start(self):
        """Start-Button: Tracker erstellen, Audio-Thread starten."""
        self.stop_event.clear()

        # Tracker (neu) erstellen
        tracker = ODTWTracker(
            reference_chroma=self.score_data.chroma,
            page_end_indices=self.score_data.page_end_indices,
            search_window=settings.SEARCH_WINDOW,
            damping_factor=settings.DAMPING_FACTOR,
            wait_penalty=settings.WAIT_PENALTY,
            skip_penalty=settings.SKIP_PENALTY,
            step_penalty=settings.STEP_PENALTY,
            page_turn_offset=settings.PAGE_TURN_OFFSET,
            start_threshold_rms=settings.START_THRESHOLD_RMS,
            smoothing_window=settings.SMOOTHING_WINDOW,
            bpm=settings.BPM,
            beats_per_measure=settings.BEATS_PER_MEASURE,
            hop_length=settings.HOP_LENGTH,
            sample_rate=settings.SAMPLE_RATE,
        )

        # Queue leeren
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except queue.Empty:
                break

        # Thread starten
        self.worker_thread = AudioProcessingThread(tracker, self.state_queue, self.stop_event)
        self.worker_thread.start()

        # UI-Status
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self._set_status("Warte auf Audio...", "#f9e2af")

        # Polling starten
        self._poll_state()

    def _on_stop(self):
        """Stop-Button: Thread stoppen, UI zurücksetzen."""
        self.stop_event.set()
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._set_status("Gestoppt", "#6c7086")

    def _poll_state(self):
        """Wird alle GUI_UPDATE_MS aufgerufen. Liest Queue und updated GUI."""
        latest_state = None

        # Alle verfügbaren States lesen, nur den letzten nutzen
        while True:
            try:
                latest_state = self.state_queue.get_nowait()
            except queue.Empty:
                break

        if latest_state is not None:
            self._update_display(latest_state)

        # Weiter pollen, solange Thread läuft
        if not self.stop_event.is_set():
            self.root.after(settings.GUI_UPDATE_MS, self._poll_state)
        elif self.worker_thread and not self.worker_thread.is_alive():
            # Thread ist fertig
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

    def _update_display(self, state: TrackerState):
        """Alle GUI-Elemente aktualisieren."""
        # Status
        if state.is_finished:
            self._set_status("Stück beendet", "#89b4fa")
            self.stop_event.set()
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
        elif state.is_running:
            self._set_status("Tracking aktiv", "#a6e3a1")
        else:
            self._set_status("Warte auf Audio...", "#f9e2af")

        # Seite
        self.page_label.configure(text=f"{state.current_page} / {state.total_pages}")

        # Fortschritt
        pct = state.progress_fraction * 100
        self.progress_bar['value'] = pct
        self.progress_label.configure(text=f"{pct:.1f} %")

        # Takt/Schlag
        self.measure_label.configure(text=f"Takt {state.measure}, Schlag {state.beat}")

        # Audio Level (0..1, logarithmisch skaliert)
        level = min(1.0, state.rms_level * 10)  # RMS ~0.01..0.1 → 0.1..1.0
        canvas_width = self.level_canvas.winfo_width() or 360
        bar_width = int(canvas_width * level)
        color = "#a6e3a1" if level < 0.6 else ("#f9e2af" if level < 0.85 else "#f38ba8")
        self.level_canvas.coords(self.level_bar, 0, 0, bar_width, 16)
        self.level_canvas.itemconfigure(self.level_bar, fill=color)

        # Cost
        self.cost_label.configure(text=f"Cost: {state.current_cost:.2f}")

        # Page Turn Flash
        if state.page_turn_triggered:
            self._flash_page_turn(state.page_turn_target)

    def _set_status(self, text: str, color: str):
        """Status-Text und -Farbe setzen."""
        self.status_label.configure(text=text)
        self.status_dot.configure(fg=color)

    def _flash_page_turn(self, page_number: int):
        """Blätter-Benachrichtigung als Flash anzeigen."""
        # Vorherigen Timer abbrechen
        if self._flash_after_id is not None:
            self.root.after_cancel(self._flash_after_id)

        self.notification_label.configure(
            text=f">>> BLÄTTERN zu Seite {page_number} <<<",
            bg="#a6e3a1",
        )

        # Nach PAGE_TURN_FLASH_MS zurücksetzen
        self._flash_after_id = self.root.after(
            settings.PAGE_TURN_FLASH_MS,
            self._clear_flash,
        )

    def _clear_flash(self):
        """Blätter-Benachrichtigung zurücksetzen."""
        self.notification_label.configure(text="", bg="#1e1e2e")
        self._flash_after_id = None


# =============================================================================
# Entry Point
# =============================================================================

def main():
    # 1. ScoreData laden
    script_dir = Path(__file__).parent
    score_path = (script_dir / settings.SCORE_DATA_PATH).resolve()

    print(f"Lade Partitur: {score_path}")
    try:
        score_data = load_score_data(str(score_path))
    except (FileNotFoundError, ValueError) as e:
        print(f"FEHLER: {e}")
        sys.exit(1)

    # 2. Audio-Geräte anzeigen
    print(f"\n--- Audio-Geräte ---")
    print(sd.query_devices())
    print(f"--------------------\n")

    # 3. GUI starten
    root = tk.Tk()
    app = PageTurnerGUI(root, score_data)
    root.mainloop()


if __name__ == "__main__":
    main()
