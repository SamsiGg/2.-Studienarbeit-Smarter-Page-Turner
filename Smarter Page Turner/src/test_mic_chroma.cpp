#include <Arduino.h>
#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include "Chroma.h"

// --- AUDIO GUI SETUP ---
// GUItool: begin automatically generated code
AudioInputI2S            i2s1;           // INMP441 Input
AudioRecordQueue         queue1;         // Warteschlange, um auf Daten zuzugreifen
AudioConnection          patchCord1(i2s1, 0, queue1, 0); // Linker Kanal -> Queue
// GUItool: end automatically generated code

// --- DSP SETUP ---
AudioDSP dsp;
int16_t audioBuffer[FFT_SIZE];
float chromaVector[NUM_CHROMA];
int bufferIndex = 0;

void setup() {
    Serial.begin(115200);
    
    // Audio Speicher reservieren (Teensy Audio Lib Standard)
    AudioMemory(30);

    // DSP Engine starten
    dsp.init();

    // Audio Input starten
    queue1.begin();
    
    Serial.println("Start Mic Test & Chroma Analysis...");
    Serial.println("C,C#,D,D#,E,F,F#,G,G#,A,A#,B"); // CSV Header
}

void loop() {
    // Wir sammeln Daten aus der Queue, bis wir FFT_SIZE (1024) Samples haben
    if (queue1.available() >= 1) {
        int16_t *buffer = queue1.readBuffer();
        
        // Daten in unseren großen Puffer kopieren
        for (int i = 0; i < 128; i++) {
            if (bufferIndex < FFT_SIZE) {
                audioBuffer[bufferIndex] = buffer[i];
                bufferIndex++;
            }
        }
        
        queue1.freeBuffer();

        // Wenn unser Puffer voll ist (1024 Samples), DSP starten
        if (bufferIndex >= FFT_SIZE) {
            
            // 1. Analyse durchführen (STFT -> Chroma)
            dsp.process(audioBuffer, chromaVector);

            // 2. CSV Ausgabe über USB Serial
            for (int i = 0; i < NUM_CHROMA; i++) {
                Serial.print(chromaVector[i], 2); // 2 Nachkommastellen
                if (i < NUM_CHROMA - 1) Serial.print(",");
            }
            Serial.println();

            // 3. Reset für nächsten Durchlauf
            // Optional: Overlap realisieren (für flüssigere Analyse), 
            // hier aber simples "Block Processing" für den Anfang.
            bufferIndex = 0; 
        }
    }
}