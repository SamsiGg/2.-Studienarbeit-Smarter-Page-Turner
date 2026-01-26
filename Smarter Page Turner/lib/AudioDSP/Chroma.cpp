#include "Chroma.h"

AudioDSP::AudioDSP() {
    // Konstruktor
}

void AudioDSP::init() {
    // Initialisiere RFFT Struktur für 1024 Punkte
    arm_rfft_fast_init_f32(&rfft_instance, FFT_SIZE);
    
    // Erstelle ein Hanning Window, um spektrale Leckage zu mindern
    for (int i = 0; i < FFT_SIZE; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (FFT_SIZE - 1)));
    }
}

float AudioDSP::getFrequency(int binIndex) {
    return (float)binIndex * SAMPLE_RATE / FFT_SIZE;
}

// Mapping von Frequenz zu Chroma (Noten C bis H)
void AudioDSP::calculateChroma(float* fftMagnitudes, float* chromaOut) {
    // Reset Chroma Vektor
    memset(chromaOut, 0, sizeof(float) * NUM_CHROMA);

    // Wir ignorieren sehr tiefe Frequenzen (unter 50Hz -> ca Bin 2)
    // und sehr hohe (über 4000Hz), da dort Musikinformation oft irrelevant ist für Page Turning
    for (int i = 2; i < FFT_SIZE / 2; i++) {
        float freq = getFrequency(i);
        float magnitude = fftMagnitudes[i];

        if (magnitude < 10.0f) continue; // Noise Gate

        // Formel: MIDI Note Number = 69 + 12 * log2(freq / 440)
        // Wir nehmen Rest 12, um die Chroma Klasse (0-11) zu bekommen
        if (freq > 0) {
            float midiNote = 69 + 12 * log2f(freq / 440.0f);
            int chromaIndex = (int(round(midiNote)) % 12);
            if (chromaIndex < 0) chromaIndex += 12;

            // Addiere Magnitude zum entsprechenden Chroma Bin
            chromaOut[chromaIndex] += magnitude;
        }
    }
    
    // Optional: Normalisierung des Vektors (Wichtig für spätere KI/DTW)
    float maxVal = 0.0f;
    for(int i=0; i<NUM_CHROMA; i++) {
        if(chromaOut[i] > maxVal) maxVal = chromaOut[i];
    }
    if(maxVal > 0.001f) {
        for(int i=0; i<NUM_CHROMA; i++) chromaOut[i] /= maxVal;
    }
}

void AudioDSP::process(int16_t* audioData, float* chromaOutput) {
    // 1. Convert Int16 zu Float und Windowing
    for (int i = 0; i < FFT_SIZE; i++) {
        fftInput[i] = (float)audioData[i] * window[i];
    }

    // 2. FFT durchführen (Real FFT)
    // arm_rfft_fast_f32 erwartet input und output buffer
    // Output ist komplex: [Real0, Img0, Real1, Img1 ...]
    arm_rfft_fast_f32(&rfft_instance, fftInput, fftOutput, 0);

    // 3. Magnitude berechnen (Betrag der komplexen Zahl)
    // Wir nutzen fftInput als temporären Puffer für Magnituden um Speicher zu sparen
    // arm_cmplx_mag_f32 schreibt die Magnituden in den Output Buffer
    float32_t magnitudes[FFT_SIZE/2];
    arm_cmplx_mag_f32(fftOutput, magnitudes, FFT_SIZE / 2);

    // 4. Chroma berechnen
    calculateChroma(magnitudes, chromaOutput);
}