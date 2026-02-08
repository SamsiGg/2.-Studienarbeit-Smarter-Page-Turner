# 🎯 ODTW Parameter Optimization Guide

## Quick Start

### 1. **Dependencies installieren**
```bash
pip install optuna plotly kaleido
```

### 2. **Optimization starten**
```bash
cd "Offline Programme/ODTW_Python"
python optimize_parameters.py
```

**Erwartete Laufzeit:** ~8-12 Stunden auf M4 Mac Mini (8 parallele Workers)

### 3. **Ergebnisse anschauen**
Nach dem Durchlauf findest du in `optimization_results/`:
- `results_TIMESTAMP.json` - Beste Parameter & Statistiken
- `optuna_study_TIMESTAMP.db` - Komplette Optuna-Datenbank
- `plot_*.html` - Interaktive Visualisierungen (in Browser öffnen)

---

## 📊 Was wird optimiert?

### Parameter
- **WAIT_PENALTY** (0.05 - 1.0): Strafe fürs Stehenbleiben
- **SKIP_PENALTY** (0.01 - 0.8): Strafe fürs Überspringen
- **DAMPING_FACTOR** (0.88 - 0.99): Dämpfung alter Kosten
- **SEARCH_WINDOW** (100 - 500): Suchfenster-Größe

### Test-Szenarien
Die Optimization testet jede Parameter-Kombination auf 9 verschiedenen Szenarien:

| Scenario | Speed | Noise | Gewicht | Beschreibung |
|----------|-------|-------|---------|--------------|
| 1 | 1.0 | 0.0 | 1.0 | Normal, kein Noise |
| 2 | 1.0 | 0.1 | 1.2 | Normal, wenig Noise |
| 3 | 1.0 | 0.3 | 1.5 | Normal, viel Noise |
| 4 | 1.3 | 0.1 | 1.3 | Schnell, wenig Noise |
| 5 | 1.3 | 0.3 | 1.8 | Schnell, viel Noise ⚡ |
| 6 | 0.7 | 0.1 | 1.3 | Langsam, wenig Noise |
| 7 | 0.7 | 0.3 | 1.8 | Langsam, viel Noise ⚡ |
| 8 | 1.5 | 0.4 | 2.0 | Extrem schnell + Noise 💀 |
| 9 | 0.5 | 0.4 | 2.0 | Extrem langsam + Noise 💀 |

**Gewichtung:** Schwierige Szenarien (hohe Gewichtung) zählen mehr für den finalen Score!

### Scoring-Funktion

Für jedes Szenario wird berechnet:

```python
score = 0.50 * accuracy       # Path MSE vom Ideal
      + 0.20 * stability      # Gleichmäßigkeit der Schritte
      + 0.15 * smoothness     # Keine Cost-Spikes
      - 0.10 * cost_penalty   # Strafe für hohe Kosten
      - 0.05 * window_penalty # Strafe für großes Window
```

**Ziel:** Hoher Score (höher = besser)

---

## 🔬 Wie funktioniert Bayesian Optimization?

1. **Initial Random Sampling** (30 Trials):
   - Erkundet den Parameter-Raum zufällig
   - Baut initiales Modell auf

2. **Tree-Parzen Estimator (TPE)**:
   - Lernt aus vorherigen Trials
   - Schlägt vielversprechende Parameter vor
   - Balanciert Exploration vs. Exploitation

3. **Pruning**:
   - Bricht schlechte Trials früh ab
   - Spart Rechenzeit

4. **Parallel Execution**:
   - 8 Workers laufen gleichzeitig
   - Nutzt alle M4 Performance-Cores

---

## 📈 Ergebnisse interpretieren

### 1. Optimization History (`plot_history_*.html`)
Zeigt Score-Verlauf über Zeit:
- **Trend nach unten** = Gut (wir minimieren negativen Score)
- **Plateau** = Konvergiert (keine Verbesserung mehr)
- Wenn nach 6h konvergiert → kann abbrechen

### 2. Parameter Importance (`plot_importance_*.html`)
Welcher Parameter hat größten Einfluss?
- **Höher** = Wichtiger für Performance
- Hilft zu verstehen, welche Parameter kritisch sind

### 3. Parallel Coordinate (`plot_parallel_*.html`)
Zeigt Zusammenhänge zwischen Parametern:
- Linien = Trials
- Farbe = Score (blau = gut, rot = schlecht)
- Muster zeigen optimale Parameter-Kombinationen

### 4. Slice Plot (`plot_slice_*.html`)
Zeigt Score vs. einzelne Parameter:
- Sweet Spots erkennen
- Trade-offs visualisieren

---

## ⚙️ Anpassungen

### Weniger Zeit? (z.B. 2 Stunden)
```python
# In optimize_parameters.py ändern:
N_TRIALS = 100          # Statt 500
N_JOBS = 8              # Bleibt gleich
TIMEOUT_HOURS = 2       # Statt 11
```

### Andere Szenarien testen?
```python
# In optimize_parameters.py ändern:
TEST_SCENARIOS = [
    (1.0, 0.0, 1.0, 'Mein Custom Scenario'),
    (1.2, 0.2, 1.5, 'Anderes Scenario'),
    # ...
]
```

### Andere Gewichtung?
```python
# In optimize_parameters.py ändern:
SCORING_WEIGHTS = {
    'accuracy': 0.60,      # Mehr Gewicht auf Accuracy
    'stability': 0.20,
    'smoothness': 0.10,
    'cost_penalty': 0.05,
    'window_penalty': 0.05
}
```

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named 'optuna'"
```bash
pip install optuna plotly kaleido
```

### Optimization ist langsam
- Reduziere `N_TRIALS`
- Reduziere `TEST_SCENARIOS` (nur wichtigste behalten)
- Erhöhe `step=50` bei `search_window` (gröbere Schritte)

### Out of Memory
- Reduziere `N_JOBS` (z.B. auf 4 statt 8)
- Kürze das Test-Audio (weniger Frames)

### Will vorzeitig abbrechen
- **Ctrl+C** drücken
- Zwischenstand wird automatisch gespeichert
- Beste Parameter werden ausgegeben

---

## 📊 Ergebnis verwenden

Nach erfolgreicher Optimization:

1. **Beste Parameter kopieren:**
   ```python
   # In dtw_engine.py:
   WAIT_PENALTY = 0.352     # Beispiel: aus Optimization
   SKIP_PENALTY = 0.187
   DAMPING_FACTOR = 0.954
   SEARCH_WINDOW = 125
   ```

2. **Validieren:**
   ```bash
   python test_robustness.py
   python dtw_engine.py  # Live-Test
   ```

3. **In Teensy übertragen:**
   ```cpp
   // In Settings.h:
   #define WAIT_PENALTY 0.352f
   #define SKIP_PENALTY 0.187f
   #define DAMPING_FACTOR 0.954f
   #define SEARCH_WINDOW 125
   ```

---

## 🎓 Weiterführendes

### Study fortsetzen (bei Abbruch)
```python
# In optimize_parameters.py:
study = optuna.load_study(
    study_name="odtw_optimization_20260207_120000",
    storage="sqlite:///optimization_results/optuna_study_20260207_120000.db"
)
study.optimize(objective, n_trials=100)  # Weitere 100 Trials
```

### Multi-Objective Optimization
Wenn du mehrere Ziele gleichzeitig optimieren willst (z.B. Accuracy UND Window-Size):
```python
study = optuna.create_study(directions=['maximize', 'minimize'])
# objective() gibt dann tuple zurück: (score, window_size)
```

---

**Viel Erfolg! 🚀**

Bei Fragen oder Problemen: Code ist ausführlich kommentiert!
