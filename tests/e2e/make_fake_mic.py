"""Writes a WAV for Chromium's fake microphone: a 0.6 s speech-like burst every 3 s, quiet room noise in between."""
import os, sys, wave
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from synth import speech_like

def main(path, silent=False, period=3.0, burst=0.6, sr=16000):
    rng = np.random.default_rng(0)
    x = 0.003 * rng.standard_normal(int(period * sr))              # a quiet room
    if not silent:
        b = speech_like(burst, sr, seed=2) * 0.6
        start = int(0.8 * sr); x[start:start + len(b)] += b
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

if __name__ == "__main__":
    main(sys.argv[1], silent="--silent" in sys.argv)
