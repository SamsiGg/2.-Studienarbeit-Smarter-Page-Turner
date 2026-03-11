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
import time
import threading
import queue
import dataclasses
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import sounddevice as sd

import settings
from score_loader import load_score_data
from chroma import AudioRingBuffer, ChromaExtractor
from dtw import ODTWTracker, TrackerState

# Notendarstellung via verovio (optional)
try:
    import io
    import ctypes.util as _ctypes_util
    import os as _os

    # cairocffi nutzt find_library, findet Homebrew-Pfade nicht → patchen
    _CAIRO_HOMEBREW_PATHS = {
        'cairo-2': '/opt/homebrew/lib/libcairo.2.dylib',
        'cairo':   '/opt/homebrew/lib/libcairo.2.dylib',
        'libcairo-2': '/opt/homebrew/lib/libcairo.2.dylib',
    }
    _orig_find_library = _ctypes_util.find_library
    def _patched_find_library(name):
        result = _orig_find_library(name)
        if result is None and name in _CAIRO_HOMEBREW_PATHS:
            candidate = _CAIRO_HOMEBREW_PATHS[name]
            if _os.path.exists(candidate):
                return candidate
        return result
    _ctypes_util.find_library = _patched_find_library

    import verovio
    import cairosvg
    from PIL import Image, ImageTk
    _VEROVIO_AVAILABLE = True
except (ImportError, OSError) as _e:
    print(f"[Notation] Import fehlgeschlagen: {_e}")
    _VEROVIO_AVAILABLE = False

# BLE Keyboard (optional – benötigt pyobjc-framework-CoreBluetooth)
try:
    from ble_keyboard import BLEKeyboard
    _BLE_AVAILABLE = True
except ImportError:
    _BLE_AVAILABLE = False
    print("[MIDI] mido oder python-rtmidi nicht installiert.")
    print("[MIDI] Installation: pip install mido python-rtmidi")


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

    def __init__(self, root: tk.Tk, score_data=None, ble=None):
        self.root = root
        self.score_data = score_data
        self.ble = ble

        # Thread-Kommunikation
        self.state_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.worker_thread: AudioProcessingThread | None = None

        # Flash-Timer
        self._flash_after_id = None
        self._recovery_after_id = None

        # Stabilitäts-Timer: letzter Zeitpunkt von Recovery/Jump
        # 0.0 → Bedingung sofort erfüllt wenn kein Recovery stattfand
        self._last_jump_time = 0.0

        self._build_ui()

        # Initial-Score anzeigen, falls beim Start bereits geladen
        if self.score_data is not None:
            self._update_score_display()

        # BLE-Status-Polling starten
        if self.ble is not None:
            self._poll_ble_status()

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
        muted = "#6c7086"

        main_frame = tk.Frame(self.root, bg=bg, padx=20, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Partitur-Auswahl ---
        score_frame = tk.Frame(main_frame, bg=surface, padx=10, pady=6)
        score_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(score_frame, text="Partitur", font=("Helvetica", 11),
                 fg=muted, bg=surface).pack(side=tk.LEFT, padx=(0, 8))

        self.score_name_label = tk.Label(
            score_frame,
            text="Keine Partitur geladen",
            font=("Helvetica", 11),
            fg=muted, bg=surface,
            anchor="w",
        )
        self.score_name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.load_btn = tk.Button(
            score_frame,
            text="Laden",
            font=("Helvetica", 11, "bold"),
            bg=accent, fg="#1e1e2e",
            activebackground="#74a8e8",
            relief=tk.FLAT,
            cursor="hand2",
            padx=12, pady=2,
            command=self._on_load_score,
        )
        self.load_btn.pack(side=tk.RIGHT)

        # --- Status ---
        status_frame = tk.Frame(main_frame, bg=surface, padx=10, pady=6)
        status_frame.pack(fill=tk.X, pady=(0, 12))

        self.status_dot = tk.Label(status_frame, text="\u25cf", font=("Helvetica", 14),
                                   fg=muted, bg=surface)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))

        self.status_label = tk.Label(status_frame, text="Bereit", font=("Helvetica", 13),
                                     fg=fg, bg=surface)
        self.status_label.pack(side=tk.LEFT)

        # BLE-Status (rechts im Status-Frame)
        if self.ble is not None:
            self.ble_status_label = tk.Label(
                status_frame, text="BLE: Startet...",
                font=("Helvetica", 11), fg=muted, bg=surface,
            )
            self.ble_status_label.pack(side=tk.RIGHT, padx=(0, 4))
            self.ble_dot = tk.Label(
                status_frame, text="\u25cf", font=("Helvetica", 12),
                fg=muted, bg=surface,
            )
            self.ble_dot.pack(side=tk.RIGHT, padx=(8, 2))
        else:
            self.ble_status_label = None
            self.ble_dot = None

        # --- Seitenanzeige ---
        page_frame = tk.Frame(main_frame, bg=bg, pady=10)
        page_frame.pack(fill=tk.X)

        tk.Label(page_frame, text="Seite", font=("Helvetica", 16), fg=muted,
                 bg=bg).pack()

        page_text = f"1 / {self.score_data.num_pages}" if self.score_data else "-- / --"
        self.page_label = tk.Label(page_frame, text=page_text,
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
                                       font=("Helvetica", 12), fg=muted, bg=bg)
        self.progress_label.pack(pady=(4, 0))

        # --- Takt/Schlag ---
        self.measure_label = tk.Label(main_frame, text="Takt 1, Schlag 1",
                                      font=("Helvetica", 15), fg=fg, bg=bg)
        self.measure_label.pack(pady=(8, 8))

        # --- Chroma-Vergleich ---
        chroma_frame = tk.Frame(main_frame, bg=bg, pady=4)
        chroma_frame.pack(fill=tk.X)

        chroma_header = tk.Frame(chroma_frame, bg=bg)
        chroma_header.pack(fill=tk.X, padx=5)
        tk.Label(chroma_header, text="Chroma", font=("Helvetica", 11),
                 fg=muted, bg=bg).pack(side=tk.LEFT)
        tk.Label(chroma_header, text="■ Referenz", font=("Helvetica", 10),
                 fg=accent, bg=bg).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Label(chroma_header, text="■ Live", font=("Helvetica", 10),
                 fg=green, bg=bg).pack(side=tk.RIGHT)

        self.chroma_canvas = tk.Canvas(chroma_frame, width=400, height=70,
                                       bg=surface, highlightthickness=0)
        self.chroma_canvas.pack(fill=tk.X, padx=5, pady=(2, 0))

        # --- Notendarstellung ---
        if _VEROVIO_AVAILABLE and settings.SHOW_NOTATION:
            notation_frame = tk.Frame(main_frame, bg=bg, pady=4)
            notation_frame.pack(fill=tk.X)
            tk.Label(notation_frame, text="Aktuelle Seite", font=("Helvetica", 11),
                     fg=muted, bg=bg).pack(anchor='w', padx=5)
            self.notation_label = tk.Label(notation_frame, bg=surface,
                                           relief=tk.FLAT, cursor="")
            self.notation_label.pack(fill=tk.X, padx=5, pady=(2, 0))
            self._notation_images: list[ImageTk.PhotoImage] = []  # Seitenbilder
            self._notation_current_page = -1
        else:
            self.notation_label = None
            self._notation_images = []
            self._notation_current_page = -1

        # --- Audio Level ---
        level_frame = tk.Frame(main_frame, bg=bg, pady=4)
        level_frame.pack(fill=tk.X)

        tk.Label(level_frame, text="Audio", font=("Helvetica", 11), fg=muted,
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

        start_state = tk.NORMAL if self.score_data else tk.DISABLED
        self.start_btn = tk.Button(btn_frame, text="START", bg=green, fg="#1e1e2e",
                                   activebackground="#94e2d5", command=self._on_start,
                                   state=start_state, **btn_style)
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

        # --- Recovery-Anzeige ---
        self.recovery_label = tk.Label(main_frame, text="",
                                       font=("Helvetica", 11, "bold"),
                                       fg="#cba6f7", bg=bg)
        self.recovery_label.pack(pady=(2, 0))

    def _on_load_score(self):
        """Dateidialog öffnen und neue Partitur laden."""
        initial_dir = Path(settings.SCORE_DATA_PATH).parent
        if not initial_dir.exists():
            initial_dir = Path.home()

        filepath = filedialog.askopenfilename(
            title="Partitur laden",
            initialdir=str(initial_dir),
            filetypes=[
                ("Partitur-Dateien", "*.npz *.h"),
                ("NumPy Archive", "*.npz"),
                ("C-Header", "*.h"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            score_data = load_score_data(filepath)
            self.score_data = score_data
            self._update_score_display()
            self._render_notation_pages()
            self.start_btn.configure(state=tk.NORMAL)
            print(f"[GUI] Partitur geladen: {filepath}")
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Fehler beim Laden",
                                 f"Partitur konnte nicht geladen werden:\n{e}")

    def _update_score_display(self):
        """Score-Name und Seitenanzeige nach dem Laden aktualisieren."""
        name = Path(self.score_data.filepath).stem
        self.score_name_label.configure(
            text=f"{name}  ({self.score_data.bpm} BPM, {self.score_data.beats_per_measure}/4)",
            fg="#cdd6f4",
        )
        self.page_label.configure(text=f"1 / {self.score_data.num_pages}")
        self.progress_bar['value'] = 0
        self.progress_label.configure(text="0.0 %")
        self.measure_label.configure(text="Takt 1, Schlag 1")
        self.cost_label.configure(text="Cost: --")

    def _on_start(self):
        """Start-Button: Tracker erstellen, Audio-Thread starten."""
        if self.score_data is None:
            return

        self.stop_event.clear()
        self._last_jump_time = 0.0   # Timer bei jedem Start zurücksetzen

        # Tracker (neu) erstellen
        page_end_indices_offset = [
            max(0, idx - settings.PAGE_TURN_OFFSET)
            for idx in self.score_data.page_end_indices
        ]
        tracker = ODTWTracker(
            reference_chroma=self.score_data.chroma,
            page_end_indices=page_end_indices_offset,
            search_window=settings.SEARCH_WINDOW,
            damping_factor=settings.DAMPING_FACTOR,
            wait_penalty=settings.WAIT_PENALTY,
            skip_penalty=settings.SKIP_PENALTY,
            step_penalty=settings.STEP_PENALTY,
            start_threshold_rms=settings.START_THRESHOLD_RMS,
            smoothing_window=settings.SMOOTHING_WINDOW,
            bpm=self.score_data.bpm,
            beats_per_measure=self.score_data.beats_per_measure,
            hop_length=settings.HOP_LENGTH,
            sample_rate=settings.SAMPLE_RATE,
            use_recovery=settings.USE_RECOVERY,
            recovery_threshold=settings.RECOVERY_THRESHOLD,
            recovery_avg_window=settings.RECOVERY_AVG_WINDOW,
            recovery_buffer_size=settings.RECOVERY_BUFFER_SIZE,
            recovery_jump_threshold=settings.RECOVERY_JUMP_THRESHOLD,
            silence_frames_before_sleep=settings.SILENCE_FRAMES_BEFORE_SLEEP,
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
        self.load_btn.configure(state=tk.DISABLED)
        self._set_status("Warte auf Audio...", "#f9e2af")

        # Polling starten
        self._poll_state()

    def _on_stop(self):
        """Stop-Button: Thread stoppen, UI zurücksetzen."""
        self.stop_event.set()
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.load_btn.configure(state=tk.NORMAL)
        self._set_status("Gestoppt", "#6c7086")

    def _poll_state(self):
        """Wird alle GUI_UPDATE_MS aufgerufen. Liest Queue und updated GUI."""
        latest_state = None

        # Einmal-Ereignisse über alle States akkumulieren, damit sie nicht
        # durch spätere States mit False-Werten überschrieben werden.
        # (Audio-Thread produziert ~4 Frames pro GUI-Poll-Intervall.)
        any_page_turn = False
        page_turn_target = 0
        any_recovered = False

        while True:
            try:
                s = self.state_queue.get_nowait()
                if s.page_turn_triggered:
                    any_page_turn = True
                    page_turn_target = s.page_turn_target
                if s.recovered:
                    any_recovered = True
                latest_state = s
            except queue.Empty:
                break

        if latest_state is not None:
            # Akkumulierte Einmal-Ereignisse in den letzten State injizieren
            latest_state = dataclasses.replace(
                latest_state,
                page_turn_triggered=any_page_turn,
                page_turn_target=page_turn_target if any_page_turn else latest_state.page_turn_target,
                recovered=any_recovered,
            )
            self._update_display(latest_state)

        # Weiter pollen, solange Thread läuft
        if not self.stop_event.is_set():
            self.root.after(settings.GUI_UPDATE_MS, self._poll_state)
        elif self.worker_thread and not self.worker_thread.is_alive():
            # Thread ist fertig
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.load_btn.configure(state=tk.NORMAL)

    def _update_display(self, state: TrackerState):
        """Alle GUI-Elemente aktualisieren."""
        # Status
        if state.is_finished:
            self._set_status("Stück beendet", "#89b4fa")
            self.stop_event.set()
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.load_btn.configure(state=tk.NORMAL)
        elif state.is_sleeping:
            self._set_status("Stille – warte auf Musik...", "#6c7086")
        elif state.is_running:
            self._set_status("Tracking aktiv", "#a6e3a1")
        else:
            self._set_status("Warte auf Audio...", "#f9e2af")

        # Seite
        self.page_label.configure(text=f"{state.current_page} / {state.total_pages}")
        # Notation: verovio hat andere Seitenaufteilung als NPZ → Fortschritt linear mappen
        if self._notation_images:
            n = len(self._notation_images)
            notation_page = max(1, min(n, int(state.progress_fraction * n) + 1))
            self._show_notation_page(notation_page)
        else:
            self._show_notation_page(state.current_page)

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

        # Chroma-Vergleich
        self._draw_chroma(state.live_chroma, state.ref_chroma)

        # Recovery Flash + Stabilitäts-Timer zurücksetzen
        if state.recovered:
            self._last_jump_time = time.time()
            self._flash_recovery(state.measure)

        # Page Turn Flash + MIDI Signal (nur wenn lange genug stabil)
        if state.page_turn_triggered:
            stable_since = time.time() - self._last_jump_time
            if stable_since >= settings.PAGE_TURN_STABLE_TIME:
                self._flash_page_turn(state.page_turn_target)
                if self.ble and self.ble.is_connected:
                    threading.Thread(target=self.ble.send_page_down,
                                     daemon=True).start()
            else:
                remaining = settings.PAGE_TURN_STABLE_TIME - stable_since
                print(f"[MIDI] Seitenumblättern blockiert – "
                      f"nur {stable_since:.1f}s stabil "
                      f"(brauche {settings.PAGE_TURN_STABLE_TIME}s, "
                      f"noch {remaining:.1f}s)")

    def _set_status(self, text: str, color: str):
        """Status-Text und -Farbe setzen."""
        self.status_label.configure(text=text)
        self.status_dot.configure(fg=color)

    def _flash_page_turn(self, page_number: int):
        """Blätter-Benachrichtigung als Flash anzeigen."""
        if self._flash_after_id is not None:
            self.root.after_cancel(self._flash_after_id)

        self.notification_label.configure(
            text=f">>> BLÄTTERN zu Seite {page_number} <<<",
            bg="#a6e3a1",
        )

        self._flash_after_id = self.root.after(
            settings.PAGE_TURN_FLASH_MS,
            self._clear_flash,
        )

    def _clear_flash(self):
        """Blätter-Benachrichtigung zurücksetzen."""
        self.notification_label.configure(text="", bg="#1e1e2e")
        self._flash_after_id = None

    def _flash_recovery(self, measure: int):
        """Recovery-Benachrichtigung anzeigen."""
        if self._recovery_after_id is not None:
            self.root.after_cancel(self._recovery_after_id)

        self.recovery_label.configure(text=f"RECOVERY → Takt {measure}")
        self._recovery_after_id = self.root.after(
            settings.RECOVERY_FLASH_MS,
            self._clear_recovery_flash,
        )

    def _clear_recovery_flash(self):
        self.recovery_label.configure(text="")
        self._recovery_after_id = None

    def _render_notation_pages(self):
        """MusicXML via verovio zu Seitenbildern vorrendern (einmalig beim Laden)."""
        self._notation_images = []
        self._notation_current_page = -1

        print(f"[Notation] verovio verfügbar: {_VEROVIO_AVAILABLE}")
        print(f"[Notation] score_data: {self.score_data is not None}")
        if self.score_data:
            content = self.score_data.musicxml_content
            print(f"[Notation] musicxml_content Länge: {len(content) if content else 0}")
            print(f"[Notation] musicxml_content Anfang: {repr(content[:80]) if content else '(leer)'}")

        if not _VEROVIO_AVAILABLE or not settings.SHOW_NOTATION or not self.score_data or not self.score_data.musicxml_content:
            if self.notation_label:
                self.notation_label.configure(image="", text="Keine MusicXML verfügbar",
                                              fg="#6c7086", font=("Helvetica", 10))
            return

        print("[Notation] Rendere Seiten mit verovio...")
        try:
            tk_obj = verovio.toolkit()
            # Breite des Notation-Widgets (Canvas-Breite ~400px)
            tk_obj.setOptions({
                "pageWidth": 800,
                "pageHeight": 400,
                "scale": 30,
                "adjustPageHeight": 1,
                "noJustification": 0,
                "breaks": "auto",
            })
            tk_obj.loadData(self.score_data.musicxml_content)
            num_pages = tk_obj.getPageCount()
            print(f"[Notation] {num_pages} Seiten erkannt.")

            images = []
            for page in range(1, num_pages + 1):
                svg_str = tk_obj.renderToSVG(page)
                png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), dpi=120)
                pil_img = Image.open(io.BytesIO(png_bytes))
                # Auf Widgetbreite skalieren
                target_w = max(390, self.notation_label.winfo_width() or 390)
                ratio = target_w / pil_img.width
                new_h = int(pil_img.height * ratio)
                pil_img = pil_img.resize((target_w, new_h), Image.LANCZOS)
                images.append(ImageTk.PhotoImage(pil_img))

            self._notation_images = images
            # Erste Seite sofort anzeigen
            if images and images[0]:
                self.notation_label.configure(image=images[0], text="")
            print("[Notation] Fertig.")
        except Exception as e:
            print(f"[Notation] Fehler beim Rendern: {e}")
            if self.notation_label:
                self.notation_label.configure(image="", text=f"Fehler: {e}",
                                              fg="#f38ba8", font=("Helvetica", 9))

    def _show_notation_page(self, page: int):
        """Seiten-Bild (1-basiert) im Notation-Label anzeigen."""
        if not self.notation_label or not self._notation_images:
            return
        idx = page - 1
        if idx < 0 or idx >= len(self._notation_images):
            return
        if page == self._notation_current_page:
            return  # kein Flackern durch doppeltes Update
        self._notation_current_page = page
        img = self._notation_images[idx]
        if img:
            self.notation_label.configure(image=img, text="")

    _NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def _draw_chroma(self, live: np.ndarray | None, ref: np.ndarray | None):
        """Chroma-Vergleich auf dem Canvas zeichnen (12 Tonklassen, 2 Balken je)."""
        c = self.chroma_canvas
        c.delete("all")

        if live is None or ref is None:
            return

        w = c.winfo_width() or 400
        h = c.winfo_height() or 70
        label_h = 14          # Platz für Noten-Namen unten
        bar_h = h - label_h  # verfügbare Balkenhöhe

        group_w = w / 12
        bar_w = max(2, group_w * 0.38)
        gap = group_w * 0.06

        # Gemeinsame Skalierung: Maximum beider Vektoren → volle Höhe
        scale = max(float(np.max(ref)), float(np.max(live)), 0.01)

        for i in range(12):
            x_center = (i + 0.5) * group_w

            # Referenz-Balken (blau, links)
            rx = x_center - gap / 2 - bar_w
            rh = (float(ref[i]) / scale) * bar_h
            c.create_rectangle(rx, bar_h - rh, rx + bar_w, bar_h,
                               fill="#89b4fa", outline="")

            # Live-Balken (grün, rechts)
            lx = x_center + gap / 2
            lh = (float(live[i]) / scale) * bar_h
            c.create_rectangle(lx, bar_h - lh, lx + bar_w, bar_h,
                               fill="#a6e3a1", outline="")

            # Noten-Name
            c.create_text(x_center, h - label_h / 2,
                          text=self._NOTE_NAMES[i],
                          font=("Helvetica", 7), fill="#6c7086")

    def _poll_ble_status(self):
        """BLE-Verbindungsstatus alle 500 ms aktualisieren."""
        if self.ble is None or self.ble_dot is None:
            return

        if self.ble.is_connected:
            self.ble_dot.configure(fg="#a6e3a1")          # grün
            self.ble_status_label.configure(text="iPad verbunden", fg="#a6e3a1")
        elif self.ble.is_advertising:
            self.ble_dot.configure(fg="#f9e2af")          # gelb
            self.ble_status_label.configure(text="MIDI: Suche...", fg="#f9e2af")
        else:
            self.ble_dot.configure(fg="#6c7086")          # grau
            self.ble_status_label.configure(text="MIDI: Startet...", fg="#6c7086")

        self.root.after(500, self._poll_ble_status)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    # 1. Standard-Partitur laden (optional – Fehler werden nicht abgebrochen)
    score_data = None
    try:
        score_path = Path(settings.SCORE_DATA_PATH).resolve()
        score_data = load_score_data(str(score_path))
    except (FileNotFoundError, ValueError) as e:
        print(f"[Startup] Keine Standard-Partitur geladen: {e}")
        print("[Startup] Bitte Partitur über die GUI auswählen.")

    # 2. Audio-Geräte anzeigen
    print(f"\n--- Audio-Geräte ---")
    print(sd.query_devices())
    print(f"--------------------\n")

    # 3. BLE Keyboard starten (optional)
    ble = None
    if settings.USE_BLE_KEYBOARD and _BLE_AVAILABLE:
        ble = BLEKeyboard()
        ble.start()
    elif settings.USE_BLE_KEYBOARD and not _BLE_AVAILABLE:
        print("[BLE] USE_BLE_KEYBOARD=True, aber pyobjc nicht installiert – BLE deaktiviert.")

    # 4. GUI starten
    root = tk.Tk()
    app = PageTurnerGUI(root, score_data, ble=ble)
    if score_data:
        root.after(200, app._render_notation_pages)  # nach erstem Layout-Pass rendern
    root.mainloop()

    # 5. Aufräumen
    if ble:
        ble.stop()


if __name__ == "__main__":
    main()
