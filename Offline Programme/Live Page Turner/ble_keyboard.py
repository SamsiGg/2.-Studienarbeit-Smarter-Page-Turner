# =============================================================================
# ble_keyboard.py – Network MIDI Page Turner (Mac → iPad über WiFi)
# =============================================================================
# Sendet MIDI-Signale über das lokale Netzwerk an ein iPad.
# Kein extra Hardware nötig – macOS hat Network MIDI eingebaut.
#
# Einrichtung (einmalig):
#   MAC:  Audio MIDI Setup.app → MIDI-Studio → Netzwerk → Eigene Sitzung aktivieren
#   iPAD: Einstellungen → Allgemein → MIDI → Netzwerk-Session → Mac auswählen
#
# Sheet-Music-App auf dem iPad:
#   → In der App MIDI-Eingang auf "Network Session" stellen
#   → Seitenumblättern auf CC 64 (oder CC 67, je nach App) konfigurieren
#
# Installation:
#   pip install mido python-rtmidi
# =============================================================================

import time
import mido


# --- MIDI-Konfiguration ---
# Control Change Nummer für Seitenumblättern.
# Gängige Belegungen:
#   CC 64 = Sustain Pedal  (forScore Standard)
#   CC 67 = Soft Pedal     (alternative)
#   CC 80 = General Purpose (Newzik)
MIDI_CC_PAGE_DOWN = 64   # Nächste Seite
MIDI_CC_PAGE_UP   = 67   # Vorherige Seite

MIDI_CHANNEL = 0         # MIDI-Kanal 1 (0-basiert)


class BLEKeyboard:
    """Network MIDI Page Turner: sendet MIDI CC-Nachrichten an ein iPad.

    Der Klassenname bleibt 'BLEKeyboard' für Kompatibilität mit main.py.
    Intern wird MIDI über das lokale Netzwerk verwendet.

    Voraussetzung: Network MIDI in Audio MIDI Setup aktiviert,
    iPad im selben WLAN und mit der Mac-Session verbunden.
    """

    def __init__(self):
        self._port = None
        self._port_name = None
        self._advertising = False   # True wenn MIDI-Port offen
        self._connected  = False    # True wenn ein MIDI-Port gefunden wurde

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_advertising(self) -> bool:
        return self._advertising

    def start(self):
        """MIDI-Port öffnen (Network Session oder ersten verfügbaren Port)."""
        available = mido.get_output_names()
        print(f"[MIDI] Verfügbare Ports: {available}")

        # Bevorzugt: Network Session (macOS Network MIDI, deutsch: "Netzwerk")
        preferred = [p for p in available if 'Network' in p or 'network' in p
                     or 'Netzwerk' in p or 'netzwerk' in p]
        if preferred:
            self._port_name = preferred[0]
        elif available:
            self._port_name = available[0]
            print(f"[MIDI] Kein Network-Port gefunden – nutze: {self._port_name}")
        else:
            print("[MIDI] FEHLER: Kein MIDI-Ausgangsport gefunden.")
            print("[MIDI] Audio MIDI Setup öffnen → Netzwerk aktivieren.")
            return

        try:
            self._port = mido.open_output(self._port_name)
            self._advertising = True
            self._connected   = True
            print(f"[MIDI] Port geöffnet: '{self._port_name}'")
            print(f"[MIDI] iPad: Einstellungen → Musik → MIDI → Netzwerk → Mac auswählen")
        except Exception as e:
            print(f"[MIDI] Port konnte nicht geöffnet werden: {e}")

    def stop(self):
        """MIDI-Port schließen."""
        if self._port:
            self._port.close()
            self._port = None
        self._advertising = False
        self._connected   = False

    def send_page_down(self):
        """Nächste Seite: MIDI CC MIDI_CC_PAGE_DOWN senden."""
        self._send_cc(MIDI_CC_PAGE_DOWN)

    def send_page_up(self):
        """Vorherige Seite: MIDI CC MIDI_CC_PAGE_UP senden."""
        self._send_cc(MIDI_CC_PAGE_UP)

    def _send_cc(self, cc_number: int):
        """Control Change Nachricht senden (Taste drücken + loslassen)."""
        if not self._port:
            return
        try:
            # CC-Wert 127 = gedrückt, 0 = losgelassen
            self._port.send(mido.Message('control_change',
                                         channel=MIDI_CHANNEL,
                                         control=cc_number,
                                         value=127))
            time.sleep(0.05)
            self._port.send(mido.Message('control_change',
                                         channel=MIDI_CHANNEL,
                                         control=cc_number,
                                         value=0))
        except Exception as e:
            print(f"[MIDI] Sendefehler: {e}")
