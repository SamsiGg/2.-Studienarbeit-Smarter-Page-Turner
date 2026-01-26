#ifndef DTW_H
#define DTW_H

#include <Arduino.h>
#include <float.h> 
#include "Settings.h"
#include "ScoreData.h"

class DTWTracker {
public:
    int current_position = 0; 
    int next_page_idx = 0;    
    bool finished = false;
    bool running = false;

    float* prev_col;
    float* curr_col;
    
    // NEU: Cache für die Längen der Vektoren
    float* score_magnitudes; 

    void init() {
        if (prev_col) delete[] prev_col;
        if (curr_col) delete[] curr_col;
        if (score_magnitudes) delete[] score_magnitudes;

        prev_col = new float[score_len];
        curr_col = new float[score_len];
        score_magnitudes = new float[score_len];

        // --- PRE-CALCULATION ---
        // Wir berechnen die Länge (Magnitude) aller Vektoren der Partitur VORHER.
        // Das spart 800 Wurzelberechnungen pro Sekunde!
        Serial.println("Pre-Calculating Score Magnitudes...");
        for (int i = 0; i < score_len; i++) {
            float dot = 0.0f;
            for (int k = 0; k < NUM_CHROMA; k++) {
                dot += score_chroma[i][k] * score_chroma[i][k];
            }
            score_magnitudes[i] = sqrt(dot);
        }

        reset();
    }

    void reset() {
        current_position = 0;
        next_page_idx = 0;
        finished = false;
        running = false;

        for (int i = 0; i < score_len; i++) {
            prev_col[i] = FLT_MAX;
            curr_col[i] = FLT_MAX;
        }
        prev_col[0] = 0.0f;
    }

    void update(float* live_chroma, float volume) {
        if (finished) return;

        if (!running) {
            if (volume > START_THRESHOLD) {
                running = true;
                Serial.println(">>> START DTW <<<");
            } else {
                return;
            }
        }

        // --- OPTIMIERUNG TEIL 2: Live Magnitude nur 1x berechnen ---
        float live_dot = 0.0f;
        for (int k = 0; k < NUM_CHROMA; k++) {
            live_dot += live_chroma[k] * live_chroma[k];
        }
        float live_mag = sqrt(live_dot);
        // Vorberechnung der Division (Multiplikation ist schneller)
        float inv_live_mag = (live_mag > 1e-9) ? (1.0f / live_mag) : 0.0f;

        // Windowing
        int start_idx = current_position - CALC_RADIUS;
        int end_idx = current_position + CALC_RADIUS;
        if (start_idx < 0) start_idx = 0;
        if (end_idx >= score_len) end_idx = score_len - 1;

        float min_val_in_col = FLT_MAX;
        int best_idx_in_col = -1;

        // --- SCHNELLE SCHLEIFE ---
        for (int j = start_idx; j <= end_idx; j++) {
            
            // 1. Distanz (Optimiert)
            // Wir machen hier KEIN sqrt mehr! Wir nutzen die vorberechneten Werte.
            float dot = 0.0f;
            for (int k = 0; k < NUM_CHROMA; k++) {
                dot += live_chroma[k] * score_chroma[j][k];
            }
            
            float score_mag = score_magnitudes[j];
            float dist = 1.0f;
            
            // Cosinus Distanz Formel: 1 - (dot / (mag1 * mag2))
            if (live_mag > 1e-9 && score_mag > 1e-9) {
                // Hier sparen wir Rechenzeit: nur Multiplikation
                float sim = dot * inv_live_mag * (1.0f / score_mag);
                if (sim > 1.0f) sim = 1.0f;
                dist = 1.0f - sim;
            }

            // 2. Kosten (Wie vorher)
            float cost_wait = prev_col[j];
            if (cost_wait < FLT_MAX) cost_wait += PENALTY_WAIT;

            float cost_step = FLT_MAX;
            if (j > 0 && prev_col[j-1] < FLT_MAX) cost_step = prev_col[j-1] + PENALTY_STEP;

            float cost_skip = FLT_MAX;
            if (j > 1 && prev_col[j-2] < FLT_MAX) cost_skip = prev_col[j-2] + PENALTY_SKIP;

            // 3. Minimum
            float min_prev = cost_wait;
            if (cost_step < min_prev) min_prev = cost_step;
            if (cost_skip < min_prev) min_prev = cost_skip;

            if (min_prev >= FLT_MAX) {
                curr_col[j] = FLT_MAX;
            } else {
                curr_col[j] = dist + min_prev;
                if (curr_col[j] < min_val_in_col) {
                    min_val_in_col = curr_col[j];
                    best_idx_in_col = j;
                }
            }
        }

        // --- PATH LOSS CHECK ---
        if (best_idx_in_col == -1 || min_val_in_col >= FLT_MAX) {
            // Pfad verloren -> Reset Buffer und Raus
             for(int k=start_idx; k<=end_idx; k++) curr_col[k] = FLT_MAX;
            return; 
        }

        // --- NORMALISIERUNG ---
        for (int j = start_idx; j <= end_idx; j++) {
            if (curr_col[j] < FLT_MAX) curr_col[j] -= min_val_in_col;
        }

        // --- SWAP ---
        float* temp = prev_col;
        prev_col = curr_col;
        curr_col = temp;
        for (int j = start_idx; j <= end_idx; j++) curr_col[j] = FLT_MAX;

        current_position = best_idx_in_col;
        checkPageTurn();
    }

private:
    void checkPageTurn() {
        if (next_page_idx < num_pages) {
            int target = page_end_indices[next_page_idx];
            if (current_position >= (target - PAGE_TURN_OFFSET)) {
                Serial1.print('n'); 
                Serial.println("\n!!! BLÄTTERN !!!\n");
                next_page_idx++;
            }
        } else {
            if (current_position >= score_len - 5) finished = true;
        }
    }
};

#endif