"""
Analyse-Tool für Optimization-Ergebnisse
-----------------------------------------
Lädt und analysiert gespeicherte Optuna-Studies.

Usage:
    python analyze_optimization.py

Author: Samuel Geffert
"""

import json
import optuna
from pathlib import Path
from datetime import datetime
import numpy as np

OUTPUT_DIR = Path("optimization_results")

def load_latest_study():
    """Lädt die neueste Optuna Study"""
    db_files = list(OUTPUT_DIR.glob("optuna_study_*.db"))

    if not db_files:
        print("❌ Keine Optuna Studies gefunden!")
        return None

    # Neueste DB finden
    latest_db = max(db_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Lade Study: {latest_db.name}")

    # Study laden
    storage = f"sqlite:///{latest_db}"
    study_name = optuna.study.get_all_study_names(storage)[0]
    study = optuna.load_study(study_name=study_name, storage=storage)

    return study

def load_latest_results():
    """Lädt die neueste JSON-Results-Datei"""
    json_files = list(OUTPUT_DIR.glob("results_*.json"))

    if not json_files:
        return None

    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest_json, 'r') as f:
        return json.load(f)

def analyze_convergence(study):
    """Analysiert Konvergenz-Verhalten"""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not trials:
        print("❌ Keine abgeschlossenen Trials!")
        return

    values = [t.value for t in trials]

    # Beste Werte über Zeit
    best_so_far = []
    current_best = float('inf')

    for v in values:
        if v < current_best:
            current_best = v
        best_so_far.append(current_best)

    # Analyse
    improvements = [i for i, (a, b) in enumerate(zip(best_so_far[:-1], best_so_far[1:])) if b < a]

    print("\n📈 Konvergenz-Analyse:")
    print(f"   Trials gesamt: {len(trials)}")
    print(f"   Verbesserungen: {len(improvements)}")

    if improvements:
        last_improvement = improvements[-1]
        print(f"   Letzte Verbesserung bei Trial: {last_improvement}")
        print(f"   Trials seit letzter Verbesserung: {len(trials) - last_improvement - 1}")

        if len(trials) - last_improvement - 1 > 50:
            print("   ⚠️  Konvergiert! (>50 Trials ohne Verbesserung)")
        else:
            print("   ✅ Noch aktiv am Optimieren")

    # Score-Statistik
    scores = [-v for v in values]  # Zurück zu positiven Scores
    print(f"\n📊 Score-Statistik:")
    print(f"   Bester Score: {max(scores):.4f}")
    print(f"   Schlechtester Score: {min(scores):.4f}")
    print(f"   Durchschnitt: {np.mean(scores):.4f}")
    print(f"   Std-Abweichung: {np.std(scores):.4f}")

def compare_top_n(study, n=5):
    """Vergleicht die Top-N Parameter-Sets"""
    trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value
    )[:n]

    if not trials:
        print("❌ Keine abgeschlossenen Trials!")
        return

    print(f"\n🏆 Top {n} Parameter-Sets:\n")

    for i, trial in enumerate(trials, 1):
        score = -trial.value
        params = trial.params

        print(f"   {i}. Score: {score:.4f}")
        print(f"      wait_penalty   = {params['wait_penalty']:.4f}")
        print(f"      skip_penalty   = {params['skip_penalty']:.4f}")
        print(f"      damping_factor = {params['damping_factor']:.4f}")
        print(f"      search_window  = {params['search_window']}")
        print()

def analyze_parameter_ranges(study):
    """Analysiert Parameter-Verteilungen der besten Trials"""
    # Beste 10% der Trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_top = max(1, len(trials) // 10)
    top_trials = sorted(trials, key=lambda t: t.value)[:n_top]

    if not top_trials:
        print("❌ Keine abgeschlossenen Trials!")
        return

    # Parameter sammeln
    params_dict = {
        'wait_penalty': [],
        'skip_penalty': [],
        'damping_factor': [],
        'search_window': []
    }

    for t in top_trials:
        for key in params_dict:
            params_dict[key].append(t.params[key])

    print(f"\n📏 Parameter-Ranges (Top {n_top} Trials = Top {100/len(trials)*n_top:.0f}%):\n")

    for param, values in params_dict.items():
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"   {param:15s}: {mean:.4f} ± {std:.4f}  [{min_val:.4f}, {max_val:.4f}]")

def suggest_refinement(study):
    """Schlägt verfeinerte Parameter-Bounds vor"""
    # Beste 20% der Trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_top = max(1, len(trials) // 5)
    top_trials = sorted(trials, key=lambda t: t.value)[:n_top]

    if not top_trials:
        return

    params_dict = {
        'wait_penalty': [],
        'skip_penalty': [],
        'damping_factor': [],
        'search_window': []
    }

    for t in top_trials:
        for key in params_dict:
            params_dict[key].append(t.params[key])

    print("\n💡 Vorschlag für verfeinerte Bounds (Feintuning):\n")
    print("```python")
    print("PARAM_BOUNDS = {")

    for param, values in params_dict.items():
        mean = np.mean(values)
        std = np.std(values)

        # Neue Bounds: ±2 Std um Mittelwert
        new_min = max(mean - 2*std, min(values))
        new_max = min(mean + 2*std, max(values))

        if param == 'search_window':
            # Auf 25er-Schritte runden
            new_min = int(np.floor(new_min / 25) * 25)
            new_max = int(np.ceil(new_max / 25) * 25)
            print(f"    '{param}': ({new_min}, {new_max}),")
        else:
            print(f"    '{param}': ({new_min:.3f}, {new_max:.3f}),")

    print("}")
    print("```")
    print("\nDann weitere 200-300 Trials mit diesen engeren Bounds!")

def main():
    """Hauptprogramm"""
    print("=" * 70)
    print("📊 Optimization Results Analyzer")
    print("=" * 70)

    # Study laden
    study = load_latest_study()
    if study is None:
        return

    results = load_latest_results()

    # Basic Info
    if results:
        print(f"\n⏱️  Optimization Time: {results['elapsed_hours']:.2f} Stunden")
        print(f"🔢 Trials: {results['n_trials']}")
        print(f"🏆 Best Score: {results['best_score']:.4f}")

    # Analysen
    analyze_convergence(study)
    compare_top_n(study, n=5)
    analyze_parameter_ranges(study)
    suggest_refinement(study)

    # Visualisierung-Erinnerung
    print("\n" + "=" * 70)
    print("💡 Tipp: Öffne die HTML-Plots im Browser für interaktive Analyse!")
    print("   optimization_results/plot_*.html")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
