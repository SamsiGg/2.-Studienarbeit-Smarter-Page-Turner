import mido, time

available = mido.get_output_names()
preferred = [p for p in available if 'Netzwerk' in p or 'Network' in p]
port_name = preferred[0] if preferred else available[0]
port = mido.open_output(port_name)
print(f"Port: {port_name}")
print("Sendet alle 2 Sekunden CC 64 (naechste Seite). Ctrl+C zum Beenden.\n")

while True:
    port.send(mido.Message('control_change', channel=0, control=64, value=127))
    port.send(mido.Message('control_change', channel=0, control=64, value=0))
    print("CC 64 gesendet")
    time.sleep(2)
