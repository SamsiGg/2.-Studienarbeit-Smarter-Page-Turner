"""
Quick Debug Test für Optimization
----------------------------------
Testet ob eine einzelne Evaluation funktioniert.
"""

import sys
from pathlib import Path

# Parent directory zu sys.path hinzufügen
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import time
from test_robustness import load_chroma_from_wav, prepare_simulation_data
from dtw_engine import load_h_file_chroma

# Test-Parameter
WAIT_PENALTY = 0.4
SKIP_PENALTY = 0.2
DAMPING_FACTOR = 0.96
SEARCH_WINDOW = 100

SCORE_FILE = "../data/ScoreData.h"
LIVE_WAV_FILE = "/Users/samuelgeffert/Desktop/GitHub/2.-Studienarbeit-Smarter-Page-Turner/Offline Programme/data/audio/Fiocco-Live (40bpm).wav"

class ParametrizedODTW:
    """ODTW mit variablen Parametern"""

    def __init__(self, reference_chroma, wait_penalty, skip_penalty,
                 damping_factor, search_window):
        self.ref = reference_chroma
        self.n_frames_ref = self.ref.shape[1]
        self.current_position_index = 0

        self.accumulated_costs = np.full(self.n_frames_ref, np.inf)
        self.accumulated_costs[0] = 0

        self.penalty_wait = wait_penalty
        self.penalty_skip = skip_penalty
        self.penalty_diagonal = 0.0
        self.damping = damping_factor
        self.search_window = int(search_window)

    def step(self, live_chroma_raw):
        """Einzelner DTW-Schritt"""
        norm = np.linalg.norm(live_chroma_raw)
        live_vec_norm = live_chroma_raw / norm if norm > 0.0001 else live_chroma_raw

        start_scan = max(0, self.current_position_index - self.search_window)
        end_scan = min(self.n_frames_ref, self.current_position_index + self.search_window)

        new_costs = np.full(self.n_frames_ref, np.inf)
        best_local_cost = np.inf
        best_index = self.current_position_index

        for i in range(start_scan, end_scan):
            dot_prod = np.clip(np.dot(live_vec_norm, self.ref[:, i]), -1.0, 1.0)
            local_dist = 1.0 - dot_prod

            cost_wait = self.accumulated_costs[i] + self.penalty_wait
            cost_step = self.accumulated_costs[i-1] + self.penalty_diagonal if i > 0 else np.inf
            cost_skip = self.accumulated_costs[i-2] + self.penalty_skip if i > 1 else np.inf

            min_prev_cost = min(cost_wait, cost_step, cost_skip)
            new_costs[i] = local_dist + (min_prev_cost * self.damping)

            if new_costs[i] < best_local_cost:
                best_local_cost = new_costs[i]
                best_index = i

        self.accumulated_costs = new_costs
        self.current_position_index = best_index

        return best_index, best_local_cost

def main():
    print("=" * 60)
    print("🔍 Debug Test: Single Evaluation")
    print("=" * 60)

    # 1. Daten laden
    print("\n1️⃣ Lade Daten...")
    try:
        start = time.time()
        ref_chroma = load_h_file_chroma(SCORE_FILE)
        print(f"   ✅ Referenz geladen: {ref_chroma.shape} ({time.time()-start:.2f}s)")

        start = time.time()
        live_chroma = load_chroma_from_wav(LIVE_WAV_FILE, save_npy=False)
        print(f"   ✅ Live Audio geladen: {live_chroma.shape} ({time.time()-start:.2f}s)")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        return

    # 2. Single Scenario Test
    print("\n2️⃣ Teste Simulation (1.0x, 0.1 Noise)...")
    try:
        start = time.time()

        # Daten vorbereiten
        live_input = prepare_simulation_data(live_chroma, 1.0, 0.1)
        print(f"   ✅ Daten vorbereitet: {live_input.shape} ({time.time()-start:.2f}s)")

        # ODTW
        start = time.time()
        dtw = ParametrizedODTW(ref_chroma, WAIT_PENALTY, SKIP_PENALTY,
                               DAMPING_FACTOR, SEARCH_WINDOW)

        positions = []
        costs = []
        n_frames = live_input.shape[1]

        for i in range(n_frames):
            live_vec = live_input[:, i]
            pos, cost = dtw.step(live_vec)
            positions.append(pos)
            costs.append(cost)

            if i % 100 == 0:
                print(f"   Frame {i}/{n_frames} ({i/n_frames*100:.1f}%)")

        elapsed = time.time() - start
        print(f"   ✅ Simulation fertig: {elapsed:.2f}s ({n_frames/elapsed:.1f} fps)")

    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Scoring
    print("\n3️⃣ Teste Scoring...")
    try:
        start = time.time()

        positions = np.array(positions)
        costs = np.array(costs)

        # MSE
        ideal_path = np.linspace(0, ref_chroma.shape[1], len(positions))
        mse = np.mean((positions - ideal_path) ** 2)

        # Score
        accuracy_score = 1.0 / (1.0 + np.sqrt(mse) / ref_chroma.shape[1])

        print(f"   ✅ Scoring fertig ({time.time()-start:.2f}s)")
        print(f"   MSE: {mse:.2f}")
        print(f"   Accuracy Score: {accuracy_score:.4f}")

    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✅ Test erfolgreich!")
    print(f"⏱️  Geschätzte Zeit pro Trial (9 Szenarien): ~{elapsed*9:.1f}s")
    print(f"⏱️  Geschätzte Zeit für 500 Trials: ~{elapsed*9*500/3600:.1f}h")
    print("=" * 60)

if __name__ == "__main__":
    main()
