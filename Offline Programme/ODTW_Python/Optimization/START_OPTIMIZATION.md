# 🚀 Schnellstart: Parameter-Optimization

## Schritt 1: Dependencies installieren

```bash
pip install optuna plotly kaleido
```

## Schritt 2: Optimization starten

```bash
cd "Offline Programme/ODTW_Python"
python optimize_parameters.py
```

**Das wars!** 🎉

---

## Was passiert jetzt?

1. **Lädt deine Daten** (ScoreData.h + Live Audio)
2. **Startet 500 Trials** mit 8 parallelen Workers
3. **Testet jede Kombination** auf 9 verschiedenen Szenarien (Speed/Noise)
4. **Findet beste Parameter** via Bayesian Optimization
5. **Speichert Ergebnisse** in `optimization_results/`

---

## Laufzeit

- **M4 Mac Mini (8 Cores):** ~8-12 Stunden
- **M1/M2 Mac:** ~10-14 Stunden
- **Intel Mac:** ~14-20 Stunden

**Tipp:** Über Nacht laufen lassen! ☕

---

## Während es läuft

Du kannst jederzeit **Ctrl+C** drücken:
- Zwischenstand wird gespeichert
- Beste bisherige Parameter werden ausgegeben
- Du kannst später fortsetzen

---

## Nach der Optimization

### 1. Ergebnisse anschauen

```bash
python analyze_optimization.py
```

Zeigt dir:
- ✅ Beste Parameter
- 📊 Statistiken
- 💡 Vorschläge für Feintuning

### 2. Visualisierungen öffnen

```bash
open optimization_results/plot_history_*.html
open optimization_results/plot_importance_*.html
```

### 3. Parameter in Code übernehmen

Kopiere die besten Werte in `dtw_engine.py`:

```python
WAIT_PENALTY = 0.352      # Beispiel aus Optimization
SKIP_PENALTY = 0.187
DAMPING_FACTOR = 0.954
SEARCH_WINDOW = 125
```

### 4. Testen!

```bash
python test_robustness.py  # Visualisierung
python dtw_engine.py       # Live-Test
```

---

## Detaillierte Anleitung

Siehe: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)

---

## Probleme?

### "ModuleNotFoundError"
```bash
pip install optuna plotly kaleido
```

### Zu langsam?
Reduziere `N_TRIALS` in `optimize_parameters.py` (Zeile 28):
```python
N_TRIALS = 100  # Statt 500 → ~2h statt 10h
```

### Out of Memory?
Reduziere `N_JOBS` in `optimize_parameters.py` (Zeile 30):
```python
N_JOBS = 4  # Statt 8
```

---

**Viel Erfolg! 🎯**
