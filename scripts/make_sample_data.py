"""Create the 1A sample clips in ../sample_audio (deterministic).

Run once:  python scripts/make_sample_data.py
The test-suite also calls build(only_missing=True) automatically, so a clone that lacks
some sample files still works.  MP3 is written with libsndfile; WebM/Opus and M4A need
ffmpeg and are skipped when it is not installed.
"""
import os, shutil, subprocess, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from synth import speech_like, tone

OUT = os.path.join(os.path.dirname(__file__), "..", "sample_audio")


def build(out=OUT, only_missing=False):
    os.makedirs(out, exist_ok=True)
    path = lambda n: os.path.join(out, n)

    def make(name, fn):
        if only_missing and os.path.exists(path(name)):
            return
        fn(path(name))

    wav = lambda x, sr: (lambda p: sf.write(p, x, sr, subtype="PCM_16"))
    s48 = speech_like(3.0, 48000)
    make("speech_like_48k_stereo.wav", wav(np.stack([s48, 0.6 * s48], 1), 48000))
    make("speech_like_44k1_mono.wav", wav(speech_like(3.0, 44100), 44100))
    make("speech_like_8k_phone.wav", wav(speech_like(3.0, 8000), 8000))
    make("speech_like_quiet_16k.wav", wav(0.01 * speech_like(3.0, 16000), 16000))
    make("speech_like_dc_offset_16k.wav", wav(0.4 * speech_like(3.0, 16000) + 0.3, 16000))
    make("speech_like_loud_clipped_16k.wav", wav(np.clip(3 * speech_like(3.0, 16000), -1, 1), 16000))
    make("silence_16k.wav", wav(np.zeros(16000, np.float32), 16000))
    make("alias_test_10k_tone_44k1.wav", wav(tone(10000, 1.0, 44100), 44100))
    make("speech_like.mp3", lambda p: sf.write(p, speech_like(3.0, 44100), 44100, format="MP3"))
    src = path("speech_like_44k1_mono.wav")
    if shutil.which("ffmpeg"):
        for name, args in [("speech_like.webm", ["-c:a", "libopus"]), ("speech_like.m4a", ["-c:a", "aac"])]:
            make(name, lambda p, a=args: subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, *a, p], check=True))
    return sorted(os.listdir(out))


if __name__ == "__main__":
    print(build())
