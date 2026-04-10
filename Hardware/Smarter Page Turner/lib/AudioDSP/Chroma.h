#ifndef CHROMA_H
#define CHROMA_H

#include <Arduino.h>
#include <arm_math.h> // Zugriff auf CMSIS-DSP des Teensy

// Konfiguration
#define FFT_SIZE 4096
#define SAMPLE_RATE 44100
#define NUM_CHROMA 12

class AudioDSP {
public:
    AudioDSP();
    void init();
    // Führt FFT durch und berechnet Chroma
    void process(int16_t* audioData, float* chromaOutput);

private:
    // Puffer für FFT Berechnungen (Complex Buffer braucht 2x Größe)
    float32_t fftInput[FFT_SIZE * 2];
    float32_t fftOutput[FFT_SIZE];
    float32_t window[FFT_SIZE];
    
    // ARM Math Instanzen
    arm_rfft_fast_instance_f32 rfft_instance;
    
    void applyWindow(int16_t* input);
    void calculateChroma(float* fftMagnitudes, float* chromaOut);
    float getFrequency(int binIndex);
};

#endif