"""Create noisy test clips for 1B/1C in ../sample_audio/noisy (deterministic)."""
import os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from synth import speech_like

SR = 16000
OUT = os.path.join(os.path.dirname(__file__), "..", "sample_audio", "noisy")


def noise(kind, n, rng):
    if kind == "white":
        y = rng.standard_normal(n)
    else:  # pink: power ~ 1/f
        X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X)); f[0] = 1
        y = np.fft.irfft(X / np.sqrt(f), n)
    return y / np.std(y)


def make_clip(snr_db, kind="white", lead=1.0, speech_s=2.5, trail=1.0, seed=0):
    """Silence + speech-like + silence, plus noise at snr_db measured against ACTIVE speech power.
    Returns (clean, noisy, t_start, t_end)."""
    rng = np.random.default_rng(seed)
    sp = speech_like(speech_s, SR, seed=1)
    clean = np.concatenate([np.zeros(int(lead * SR)), sp, np.zeros(int(trail * SR))]).astype(np.float64)
    nz = noise(kind, len(clean), rng) * np.sqrt(np.mean(sp.astype(float) ** 2) / 10 ** (snr_db / 10))
    return clean.astype(np.float32), (clean + nz).astype(np.float32), lead, lead + speech_s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    clean, _, _, _ = make_clip(100)
    sf.write(os.path.join(OUT, "clean_reference.wav"), clean, SR, subtype="PCM_16")
    for kind in ("white", "pink"):
        for snr in (20, 10, 5):
            _, noisy, _, _ = make_clip(snr, kind)
            sf.write(os.path.join(OUT, f"{kind}_{snr}dB.wav"), noisy / max(1, np.abs(noisy).max()), SR, subtype="PCM_16")
    print(sorted(os.listdir(OUT)))
