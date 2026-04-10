#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>

#include "Settings.h"
#include "Chroma.h"      
#include "DTW.h"         
#include "ScoreData.h"   

AudioInputI2S            i2s1;           
AudioRecordQueue         queue1;         
AudioConnection          patchCord1(i2s1, 0, queue1, 0); 

AudioDSP dsp;            
DTWTracker tracker;      

int16_t audioBuffer[FFT_SIZE];
float chromaVector[NUM_CHROMA];
int bufferIndex = 0;

void setup() {
    Serial.begin(115200);
    Serial1.begin(9600); 
    AudioMemory(60); 

    dsp.init();
    tracker.init();
    queue1.begin();
    
    delay(1000);
    Serial.println("System Bereit. Warte auf Audio...");
}

void loop() {
    // 1. Audio sammeln
    if (queue1.available() >= 1) {
        int16_t *buffer = queue1.readBuffer();
        
        for (int i = 0; i < 128; i++) {
            if (bufferIndex < FFT_SIZE) {
                audioBuffer[bufferIndex] = buffer[i];
                bufferIndex++;
            }
        }
        queue1.freeBuffer();

        // 2. Wenn Puffer voll -> Verarbeiten
        if (bufferIndex >= FFT_SIZE) {
            
            // --- ZEITSTEMPEL HOLEN ---
            float timestamp = millis() / 1000.0;

            // Lautstärke berechnen
            float sum = 0;
            for(int i=0; i<FFT_SIZE; i++) sum += abs(audioBuffer[i]);
            float volume = sum / FFT_SIZE;

            // Berechnung
            dsp.process(audioBuffer, chromaVector);
            tracker.update(chromaVector, volume);

            // --- AUSGABE JEDEN FRAME ---
            // Nur ausgeben, wenn Tracker läuft (sonst spammt er "Waiting")
            if (tracker.running) {
                Serial.print("["); 
                Serial.print(timestamp, 3); // 3 Nachkommastellen (ms)
                Serial.print("s] ");
                
                // Fortschrittsbalken
                int barWidth = 20;
                float progress = (float)tracker.current_position / (float)score_len;
                int pos = barWidth * progress;
                
                Serial.print("[");
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) Serial.print("=");
                    else if (i == pos) Serial.print(">");
                    else Serial.print(" ");
                }
                Serial.print("] Pos: ");
                Serial.print(tracker.current_position);
                
                // Debug Kosten (vom aktuellen Frame)
                // Wir greifen direkt auf das Array im Tracker zu
                // Vorsicht: prev_col enthält jetzt die Werte dieses Durchlaufs (wegen swap am Ende von update)
                if (tracker.current_position < score_len) {
                    Serial.print(" | Cost: ");
                    Serial.println(tracker.prev_col[tracker.current_position], 2);
                } else {
                    Serial.println();
                }
            }

            bufferIndex = 0; 
        }
    }
}