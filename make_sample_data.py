"""Create the sample clips in ../sample_audio (run once; deterministic)."""
import os, subprocess, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from synth import speech_like, tone

out = os.path.join(os.path.dirname(__file__), "..", "sample_audio")
os.makedirs(out, exist_ok=True)
w = lambda name, x, sr, sub="PCM_16": sf.write(os.path.join(out, name), x, sr, subtype=sub)

s48 = speech_like(3.0, 48000)
w("speech_like_48k_stereo.wav", np.stack([s48, 0.6 * s48], 1), 48000)          # 2 ch, 48 kHz
w("speech_like_44k1_mono.wav", speech_like(3.0, 44100), 44100)
w("speech_like_8k_phone.wav", speech_like(3.0, 8000), 8000)                    # telephone rate
w("speech_like_quiet_16k.wav", 0.01 * speech_like(3.0, 16000), 16000)          # -40 dBFS
w("speech_like_dc_offset_16k.wav", 0.4 * speech_like(3.0, 16000) + 0.3, 16000) # DC +0.3
w("speech_like_loud_clipped_16k.wav", np.clip(3 * speech_like(3.0, 16000), -1, 1), 16000)
w("silence_16k.wav", np.zeros(16000, np.float32), 16000)
w("alias_test_10k_tone_44k1.wav", tone(10000, 1.0, 44100), 44100)
src = os.path.join(out, "speech_like_44k1_mono.wav")
for name, args in [("speech_like.mp3", ["-c:a", "libmp3lame"]),
                   ("speech_like.webm", ["-c:a", "libopus"]),
                   ("speech_like.m4a", ["-c:a", "aac"])]:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, *args,
                    os.path.join(out, name)], check=True)
print(sorted(os.listdir(out)))
