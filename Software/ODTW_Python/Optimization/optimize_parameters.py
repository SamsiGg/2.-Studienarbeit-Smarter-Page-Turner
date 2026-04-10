"""
ODTW Parameter Optimization mit Bayesian Optimization
------------------------------------------------------
Nutzt Optuna (Tree-Parzen Estimator) um optimale Parameter zu finden für:
- WAIT_PENALTY
- SKIP_PENALTY
- DAMPING_FACTOR
- SEARCH_WINDOW

Optimiert auf mehreren Szenarien (Speed/Noise-Kombinationen) für Robustheit.

Benötigt: pip install optuna plotly kaleido

Author: Samuel Geffert
Laufzeit: ~8-12 Stunden auf M4 Mac Mini (8 parallele Workers)
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
import json
import time
from datetime import datetime
import numpy as np

# Importiere deine existierenden Module
from test_robustness import (
    load_chroma_from_wav,
    prepare_simulation_data,
    LIVE_WAV_FILE,
    SAMPLE_RATE,
    HOP_LENGTH
)
from dtw_engine import load_h_file_chroma

# --- KONFIGURATION ---
SCORE_FILE = "../data/ScoreData.h"
OUTPUT_DIR = Path("optimization_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Optuna Settings
N_TRIALS = 500           # Anzahl Parameter-Kombinationen
N_STARTUP_TRIALS = 30    # Initial random sampling
N_JOBS = 8               # Parallel workers (M4 hat 10 Cores, 8 nutzen)
TIMEOUT_HOURS = 11       # Max Laufzeit

# Test-Szenarien (Speed, Noise, Gewichtung)
# Gewichtung: Höher = wichtiger für finalen Score
TEST_SCENARIOS = [
    (1.0, 0.0, 1.0, 'Normal - Kein Noise'),
    (1.0, 0.3, 1.5, 'Normal - Viel Noise'),
    (1.3, 0.1, 1.3, 'Schnell - Wenig Noise'),
    (1.3, 0.3, 2.0, 'Schnell - Viel Noise'),
    (0.7, 0.3, 2.0, 'Langsam - Viel Noise'),
]

# Parameter-Bounds
PARAM_BOUNDS = {
    'wait_penalty': (0.05, 1.0),
    'skip_penalty': (0.01, 0.8),
    'damping_factor': (0.88, 0.99),
    'search_window': (100, 500)
}

# Scoring-Gewichte
SCORING_WEIGHTS = {
    'accuracy': 0.50,      # Path-Genauigkeit (wichtigster Faktor)
    'stability': 0.20,     # Gleichmäßigkeit der Bewegung
    'smoothness': 0.15,    # Keine Cost-Spikes
    'cost_penalty': 0.10,  # Strafe für hohe Kosten
    'window_penalty': 0.05 # Strafe für großes Search Window
}

# --- KLASSEN ---

class ParametrizedODTW:
    """ODTW mit variablen Parametern für Optimization"""

    def __init__(self, reference_chroma, wait_penalty, skip_penalty,
                 damping_factor, search_window):
        self.ref = reference_chroma
        self.n_frames_ref = self.ref.shape[1]
        self.current_position_index = 0

        self.accumulated_costs = np.full(self.n_frames_ref, np.inf)
        self.accumulated_costs[0] = 0

        # Parametrisiert
        self.penalty_wait = wait_penalty
        self.penalty_skip = skip_penalty
        self.penalty_diagonal = 0.0
        self.damping = damping_factor
        self.search_window = int(search_window)

    def step(self, live_chroma_raw):
        """Einzelner DTW-Schritt"""
        # Normalisierung
        norm = np.linalg.norm(live_chroma_raw)
        live_vec_norm = live_chroma_raw / norm if norm > 0.0001 else live_chroma_raw

        # Suchfenster
        start_scan = max(0, self.current_position_index - self.search_window)
        end_scan = min(self.n_frames_ref, self.current_position_index + self.search_window)

        new_costs = np.full(self.n_frames_ref, np.inf)
        best_local_cost = np.inf
        best_index = self.current_position_index

        for i in range(start_scan, end_scan):
            # Lokale Distanz (Cosine)
            dot_prod = np.clip(np.dot(live_vec_norm, self.ref[:, i]), -1.0, 1.0)
            local_dist = 1.0 - dot_prod

            # Pfad-Auswahl
            cost_wait = self.accumulated_costs[i] + self.penalty_wait
            cost_step = self.accumulated_costs[i-1] + self.penalty_diagonal if i > 0 else np.inf
            cost_skip = self.accumulated_costs[i-2] + self.penalty_skip if i > 1 else np.inf

            min_prev_cost = min(cost_wait, cost_step, cost_skip)

            # Gesamtkosten
            new_costs[i] = local_dist + (min_prev_cost * self.damping)

            if new_costs[i] < best_local_cost:
                best_local_cost = new_costs[i]
                best_index = i

        self.accumulated_costs = new_costs
        self.current_position_index = best_index

        return best_index, best_local_cost

# --- EVALUATION ---

def run_single_scenario(ref_chroma, live_chroma, wait_penalty, skip_penalty,
                       damping_factor, search_window, speed, noise):
    """
    Führt eine Simulation durch und gibt Tracking-Daten zurück.

    Returns:
        positions: Liste der erkannten Frame-Positionen
        costs: Liste der globalen Kosten
    """
    # Daten vorbereiten (Speed + Noise)
    live_input = prepare_simulation_data(live_chroma, speed, noise)

    # ODTW Engine mit Parametern
    dtw = ParametrizedODTW(ref_chroma, wait_penalty, skip_penalty,
                           damping_factor, search_window)

    positions = []
    costs = []

    # Simulation
    for i in range(live_input.shape[1]):
        live_vec = live_input[:, i]
        pos, cost = dtw.step(live_vec)
        positions.append(pos)
        costs.append(cost)

    return np.array(positions), np.array(costs)

def calculate_scenario_score(positions, costs, ref_len):
    """
    Berechnet Score für ein einzelnes Szenario.

    Returns:
        dict mit einzelnen Score-Komponenten
    """
    n_frames = len(positions)
    ideal_path = np.linspace(0, ref_len, n_frames)

    # 1. Path Accuracy (MSE vom Ideal)
    mse = np.mean((positions - ideal_path) ** 2)
    # Normalisieren: Bei perfektem Tracking = 1.0, schlechter = niedriger
    accuracy_score = 1.0 / (1.0 + np.sqrt(mse) / ref_len)

    # 2. Stability (Gleichmäßigkeit der Schritte)
    position_jumps = np.diff(positions)
    # Ideale Schrittgröße
    ideal_jump = ref_len / n_frames
    jump_variance = np.var(position_jumps - ideal_jump)
    stability_score = 1.0 / (1.0 + jump_variance)

    # 3. Cost Smoothness (Keine wilden Spikes)
    if len(costs) > 1:
        cost_changes = np.abs(np.diff(costs))
        cost_variance = np.var(cost_changes)
        smoothness_score = 1.0 / (1.0 + cost_variance)
    else:
        smoothness_score = 1.0

    # 4. Cost Penalty (Hohe Kosten = schlecht)
    max_cost = np.max(costs)
    mean_cost = np.mean(costs)
    # Strafe wenn Kosten zu hoch (Schwellwert: 10.0)
    cost_penalty = max(0, max_cost - 10.0) + max(0, mean_cost - 5.0)

    return {
        'accuracy': accuracy_score,
        'stability': stability_score,
        'smoothness': smoothness_score,
        'cost_penalty': cost_penalty,
        'max_cost': max_cost,
        'mean_cost': mean_cost,
        'mse': mse
    }

def evaluate_parameters(ref_chroma, live_chroma, wait_penalty, skip_penalty,
                       damping_factor, search_window):
    """
    Evaluiert ein Parameter-Set auf allen Test-Szenarien.

    Returns:
        weighted_score: Finaler gewichteter Score (höher = besser)
        details: Dict mit detaillierten Ergebnissen
    """
    ref_len = ref_chroma.shape[1]
    total_weighted_score = 0.0
    total_weight = 0.0
    scenario_results = []

    for speed, noise, weight, label in TEST_SCENARIOS:
        # Simulation
        positions, costs = run_single_scenario(
            ref_chroma, live_chroma,
            wait_penalty, skip_penalty, damping_factor, search_window,
            speed, noise
        )

        # Scoring
        scores = calculate_scenario_score(positions, costs, ref_len)

        # Gewichtete Kombination der Score-Komponenten
        scenario_score = (
            SCORING_WEIGHTS['accuracy'] * scores['accuracy'] +
            SCORING_WEIGHTS['stability'] * scores['stability'] +
            SCORING_WEIGHTS['smoothness'] * scores['smoothness'] -
            SCORING_WEIGHTS['cost_penalty'] * scores['cost_penalty']
        )

        # Szenario-Gewichtung anwenden (schwierige Szenarien zählen mehr)
        weighted_scenario_score = scenario_score * weight
        total_weighted_score += weighted_scenario_score
        total_weight += weight

        scenario_results.append({
            'label': label,
            'speed': speed,
            'noise': noise,
            'weight': weight,
            'scenario_score': scenario_score,
            'weighted_score': weighted_scenario_score,
            **scores
        })

    # Durchschnitt über alle Szenarien
    avg_score = total_weighted_score / total_weight

    # Window Penalty (kleinere Windows bevorzugen)
    window_penalty = (search_window - 100) / 400  # 0 bei 100, 1 bei 500
    final_score = avg_score - SCORING_WEIGHTS['window_penalty'] * window_penalty

    details = {
        'final_score': final_score,
        'avg_scenario_score': avg_score,
        'window_penalty': window_penalty,
        'scenarios': scenario_results
    }

    return final_score, details

# --- OPTUNA OBJECTIVE ---

# Globale Variablen für Objective (werden in main() gesetzt)
_ref_chroma = None
_live_chroma = None

def objective(trial):
    """Optuna Objective Function"""
    global _ref_chroma, _live_chroma

    # Parameter vorschlagen
    wait_penalty = trial.suggest_float('wait_penalty', *PARAM_BOUNDS['wait_penalty'])
    skip_penalty = trial.suggest_float('skip_penalty', *PARAM_BOUNDS['skip_penalty'])
    damping_factor = trial.suggest_float('damping_factor', *PARAM_BOUNDS['damping_factor'])
    search_window = trial.suggest_int('search_window', *PARAM_BOUNDS['search_window'], step=25)

    # Evaluieren
    try:
        score, details = evaluate_parameters(
            _ref_chroma, _live_chroma,
            wait_penalty, skip_penalty, damping_factor, search_window
        )

        # User Attributes für detaillierte Analyse
        trial.set_user_attr('avg_scenario_score', details['avg_scenario_score'])
        trial.set_user_attr('window_penalty', details['window_penalty'])

        # Optuna minimiert, wir wollen maximieren → negieren
        return -score

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')  # Schlechter Score bei Fehler

# --- MAIN ---

def main():
    """Hauptprogramm"""
    global _ref_chroma, _live_chroma

    print("=" * 70)
    print("🎯 ODTW Parameter Optimization mit Bayesian Optimization")
    print("=" * 70)

    # Daten laden
    print("\n📂 Lade Daten...")
    try:
        _ref_chroma = load_h_file_chroma(SCORE_FILE)
        _live_chroma = load_chroma_from_wav(LIVE_WAV_FILE, save_npy=False)
        print(f"✅ Referenz: {_ref_chroma.shape[1]} Frames")
        print(f"✅ Live Audio: {_live_chroma.shape[1]} Frames")
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return

    # Optuna Study erstellen
    print(f"\n🔬 Erstelle Optuna Study...")
    print(f"   Trials: {N_TRIALS}")
    print(f"   Parallel Workers: {N_JOBS}")
    print(f"   Timeout: {TIMEOUT_HOURS}h")
    print(f"   Test-Szenarien: {len(TEST_SCENARIOS)}")

    study = optuna.create_study(
        study_name=f"odtw_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction='minimize',  # Wir negieren den Score in objective()
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=N_STARTUP_TRIALS,
            multivariate=True,
            seed=42  # Reproduzierbarkeit
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=3
        )
    )

    # Optimization starten
    print(f"\n🚀 Starte Optimization... (Das dauert ~{TIMEOUT_HOURS}h)")
    print("   Drücke Ctrl+C um vorzeitig zu beenden (Zwischenstand wird gespeichert)\n")

    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            timeout=TIMEOUT_HOURS * 3600,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization abgebrochen (Zwischenstand wird gespeichert)")

    elapsed_time = time.time() - start_time

    # Ergebnisse
    print("\n" + "=" * 70)
    print("✅ Optimization abgeschlossen!")
    print("=" * 70)
    print(f"\n⏱️  Laufzeit: {elapsed_time/3600:.2f} Stunden")
    print(f"🔢 Trials durchgeführt: {len(study.trials)}")
    print(f"🏆 Beste Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    # Beste Parameter
    best_params = study.best_params
    best_score = -study.best_value  # Zurück-negieren

    print(f"\n🎯 **BESTE PARAMETER** (Score: {best_score:.4f}):")
    print(f"   WAIT_PENALTY    = {best_params['wait_penalty']:.4f}")
    print(f"   SKIP_PENALTY    = {best_params['skip_penalty']:.4f}")
    print(f"   DAMPING_FACTOR  = {best_params['damping_factor']:.4f}")
    print(f"   SEARCH_WINDOW   = {best_params['search_window']}")

    # Speichern
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. JSON Results
    results = {
        'timestamp': timestamp,
        'elapsed_hours': elapsed_time / 3600,
        'n_trials': len(study.trials),
        'best_score': best_score,
        'best_params': best_params,
        'param_bounds': PARAM_BOUNDS,
        'test_scenarios': [
            {'speed': s, 'noise': n, 'weight': w, 'label': l}
            for s, n, w, l in TEST_SCENARIOS
        ],
        'scoring_weights': SCORING_WEIGHTS
    }

    json_file = OUTPUT_DIR / f"results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Ergebnisse gespeichert: {json_file}")

    # 2. Optuna Database
    db_file = OUTPUT_DIR / f"optuna_study_{timestamp}.db"
    study_copy = optuna.copy_study(
        from_study_name=study.study_name,
        from_storage=study._storage,
        to_storage=f"sqlite:///{db_file}"
    )
    print(f"💾 Optuna DB gespeichert: {db_file}")

    # 3. Visualisierungen
    print("\n📊 Erstelle Visualisierungen...")

    try:
        # Optimization History
        fig = plot_optimization_history(study)
        fig.write_html(OUTPUT_DIR / f"plot_history_{timestamp}.html")
        print(f"   ✅ Optimization History")

        # Parameter Importance
        fig = plot_param_importances(study)
        fig.write_html(OUTPUT_DIR / f"plot_importance_{timestamp}.html")
        print(f"   ✅ Parameter Importance")

        # Parallel Coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_html(OUTPUT_DIR / f"plot_parallel_{timestamp}.html")
        print(f"   ✅ Parallel Coordinate")

        # Slice Plot
        fig = plot_slice(study)
        fig.write_html(OUTPUT_DIR / f"plot_slice_{timestamp}.html")
        print(f"   ✅ Slice Plot")

    except Exception as e:
        print(f"   ⚠️  Visualisierung fehlgeschlagen: {e}")

    # 4. Beste N Parameter ausgeben
    print("\n📈 Top 5 Parameter-Sets:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))[:5]

    for i, trial in enumerate(top_trials, 1):
        if trial.value is None:
            continue
        score = -trial.value
        print(f"\n   {i}. Score: {score:.4f}")
        print(f"      wait={trial.params['wait_penalty']:.3f}, "
              f"skip={trial.params['skip_penalty']:.3f}, "
              f"damp={trial.params['damping_factor']:.3f}, "
              f"window={trial.params['search_window']}")

    print("\n" + "=" * 70)
    print("🎉 Fertig! Kopiere die besten Parameter in dtw_engine.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
