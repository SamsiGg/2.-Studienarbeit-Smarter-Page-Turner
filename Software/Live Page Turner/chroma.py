# =============================================================================
# chroma.py – Audio → Chroma-Extraktion
# =============================================================================
# Teensy-Äquivalent: Chroma.h / Chroma.cpp (AudioDSP)
#   - Teensy: ARM CMSIS-DSP FFT + manuelle Frequenz→Chroma Zuordnung
#   - Python: librosa.feature.chroma_stft (gleiche Logik, andere Implementierung)
#
# Beide Ansätze:
#   1. Hanning-Fenster anwenden
#   2. FFT-Magnitudenspektrum berechnen
#   3. Frequenz-Bins auf 12 Chroma-Bins (C, C#, D, ..., B) abbilden
# =============================================================================

import numpy as np
import librosa


class AudioRingBuffer:
    """Ringpuffer für überlappende FFT-Frames.

    Teensy-Äquivalent: Der Audio-Buffer in odtw_turner.cpp, der 128-Sample-Blöcke
    sammelt bis BLOCK_SIZE erreicht ist. Hier nutzen wir HOP_LENGTH-basiertes
    Sliding für höhere Update-Rate (~86 Hz statt ~10 Hz).

    Args:
        size: Puffergröße in Samples (= BLOCK_SIZE).
    """

    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)

    def append(self, new_data: np.ndarray):
        """Neue Samples anfügen, alte nach links schieben."""
        n = len(new_data)
        self.buffer = np.roll(self.buffer, -n)
        self.buffer[-n:] = new_data

    def get(self) -> np.ndarray:
        """Gesamten Pufferinhalt für FFT zurückgeben."""
        return self.buffer


class ChromaExtractor:
    """Berechnet 12-dimensionale Chroma-Vektoren aus Audio-Buffern.

    Teensy-Äquivalent: AudioDSP::process() in Chroma.cpp
      - Dort: arm_rfft_fast_f32 + manuelle Bin→MIDI→Chroma Zuordnung
      - Hier: librosa.feature.chroma_stft (gleiche Mathematik, NumPy/SciPy FFT)

    Args:
        sample_rate: Audio-Samplerate.
        block_size: FFT-Fenstergröße.
    """

    def __init__(self, sample_rate: int, block_size: int):
        self.sample_rate = sample_rate
        self.block_size = block_size

    def extract(self, audio_buffer: np.ndarray) -> np.ndarray:
        """Chroma-Vektor aus Audio-Buffer extrahieren.

        Args:
            audio_buffer: Float-Array der Länge block_size.

        Returns:
            Chroma-Vektor, Shape (12,), unnormalisiert (DTW normalisiert selbst).
        """
        chroma = librosa.feature.chroma_stft(
            y=audio_buffer,
            sr=self.sample_rate,
            n_fft=self.block_size,
            hop_length=self.block_size + 1,  # Trick: nur 1 Frame berechnen
            center=False,
            tuning=0,  # Kein Auto-Tuning (wie auf dem Teensy)
        )
        # librosa gibt (12, 1) zurück → flatten zu (12,)
        return chroma[:, 0]

    @staticmethod
    def compute_rms(audio_buffer: np.ndarray) -> float:
        """RMS-Energie berechnen (für Audio-Level-Anzeige).

        Teensy-Äquivalent: Volume-Berechnung in odtw_turner.cpp.
        """
        return float(np.sqrt(np.mean(audio_buffer ** 2)))
