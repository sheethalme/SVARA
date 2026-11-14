"""Deterministic synthetic test signals (no downloads, no randomness beyond a fixed seed).

``speech_like`` imitates the structure of voiced speech: a glottal-pulse harmonic source
(F0 glide 110->140 Hz) shaped by three formant resonances (700/1200/2600 Hz, i.e. an
open vowel), gated into syllable-rate bursts with pauses.  It is NOT real speech; it is a
controlled signal whose spectrum we know exactly, which is what 1A's resampling and
level-handling tests need.
"""
import numpy as np
from scipy.signal import lfilter


def tone(freq, dur, sr, amp=0.5, phase=0.0):
    t = np.arange(int(round(dur * sr))) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def speech_like(dur=3.0, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = 110 + 30 * t / dur
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = sum(np.sin(k * phase) / k for k in range(1, int(sr / 2 / 140)))  # harmonics up to Nyquist
    y = np.zeros(n)
    for fc, bw in [(700, 110), (1200, 120), (2600, 160)]:            # formant resonators
        r = np.exp(-np.pi * bw / sr)
        a = [1, -2 * r * np.cos(2 * np.pi * fc / sr), r * r]
        y += lfilter([1 - r], a, src)
    gate = (np.sin(2 * np.pi * 3.0 * t) > -0.2).astype(float)         # ~3 syllables/s
    gate = np.convolve(gate, np.hanning(401) / np.hanning(401).sum(), "same")
    y = y * gate + 0.002 * rng.standard_normal(n)
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32)
