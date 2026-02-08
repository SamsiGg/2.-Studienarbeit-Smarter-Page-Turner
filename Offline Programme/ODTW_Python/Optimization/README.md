# ODTW Parameter Optimization

Dieses Verzeichnis enthält Tools zur automatischen Optimierung der ODTW-Parameter mittels Bayesian Optimization.

## 📁 Dateien

### Hauptskripte
- **`optimize_parameters.py`** - Hauptskript für die Bayesian Optimization (Optuna)
- **`analyze_optimization.py`** - Analysiert die Ergebnisse einer abgeschlossenen Optimization
- **`test_optimization.py`** - Debug-Test für einzelne Evaluationen

### Dokumentation
- **`OPTIMIZATION_GUIDE.md`** - Detaillierte technische Dokumentation
- **`START_OPTIMIZATION.md`** - Quick-Start-Anleitung
- **`README.md`** - Diese Datei

### Ergebnisse (wird automatisch erstellt)
- **`optimization_results/`** - Ordner für Optuna-Datenbanken, JSON-Results und HTML-Plots

## 🚀 Quick Start

```bash
# Optimization starten (dauert mehrere Stunden!)
python optimize_parameters.py

# Ergebnisse analysieren
python analyze_optimization.py

# Single Trial Debug-Test
python test_optimization.py
```

## 📊 Zu optimierende Parameter

| Parameter | Bereich | Beschreibung |
|-----------|---------|--------------|
| `wait_penalty` | 0.05 - 1.0 | Strafe für Warten auf gleicher Position |
| `skip_penalty` | 0.01 - 0.8 | Strafe für Überspringen von Frames |
| `damping_factor` | 0.88 - 0.99 | Dämpfung vergangener Kosten (wichtigster Parameter!) |
| `search_window` | 100 - 500 | Suchfenster-Größe um aktuelle Position |

## ⚠️ Wichtige Erkenntnisse

Nach empirischen Tests hat sich gezeigt:
- **Automatische Optimization ist nicht immer besser als manuelle Tests**
- Score-Funktion erfasst nicht alle realen Anwendungsfälle
- Synthetische Test-Daten ≠ echte Audio-Aufnahmen
- **Damping Factor** ist mit Abstand der wichtigste Parameter (Importance ~0.99)
- Empirisch gefundene optimale Werte:
  ```python
  DAMPING_FACTOR = 0.96
  WAIT_PENALTY = 0.4
  SKIP_PENALTY = 0.2
  SEARCH_WINDOW = 100
  ```

## 🔗 Siehe auch

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Vollständige technische Dokumentation
- [START_OPTIMIZATION.md](START_OPTIMIZATION.md) - Detaillierte Start-Anleitung
- `../dtw_engine.py` - ODTW-Implementierung mit aktuellen optimalen Parametern
- `../test_robustness.py` - Robustness-Tests für verschiedene Szenarien
