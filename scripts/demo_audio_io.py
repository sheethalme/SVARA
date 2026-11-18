"""Produces the figures + results table used in docs/. Run: python scripts/demo_audio_io.py"""
import os, sys, csv
import numpy as np, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from audio_io import load_audio_with_report
from synth import tone
S = os.path.join(ROOT, "sample_audio"); F = os.path.join(ROOT, "docs", "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure 1: before / after for a hostile input (48 kHz stereo)
name = "speech_like_dc_offset_16k.wav"
raw, sr0 = sf.read(os.path.join(S, "speech_like_48k_stereo.wav"), always_2d=True)
y, sr, rep = load_audio_with_report(os.path.join(S, "speech_like_48k_stereo.wav"))
fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True, sharey=True)
t0 = np.arange(len(raw)) / sr0; t1 = np.arange(len(y)) / sr
ax[0].plot(t0, raw[:, 0], lw=.4, label="left"); ax[0].plot(t0, raw[:, 1], lw=.4, label="right")
ax[0].set_title(f"Input: {sr0} Hz, 2 channels, peak {np.abs(raw).max():.2f}"); ax[0].legend(loc="lower right", ncol=2, fontsize=7)
ax[1].plot(t1, y, lw=.4, color="C2"); ax[1].set_title(f"Output of load_audio(): {sr} Hz, mono, peak {np.abs(y).max():.2f}")
ax[1].set_xlabel("time (s)"); [a.set_ylim(-1, 1) for a in ax]; fig.tight_layout()
fig.savefig(os.path.join(F, "fig1_before_after.png")); plt.close(fig)

# ---- Figure 2: anti-aliasing. 10 kHz tone @44.1k -> 16k
x = tone(10000, 1.0, 44100, 0.5)
naive = x[::3][:len(x)//3]            # keep every 3rd sample -> 14.7 kHz (no low-pass!), aliased
naive16 = np.interp(np.arange(16000)/16000, np.arange(len(x))/44100, x)  # linear interp, no LPF
good, _ = None, None
from audio_io import resample_audio
good = resample_audio(x, 44100, 16000)
def spec(sig, sr):
    s = 20*np.log10(np.abs(np.fft.rfft(sig*np.hanning(len(sig))))/ (len(sig)/4) + 1e-9)
    return np.fft.rfftfreq(len(sig), 1/sr), s
fig, ax = plt.subplots(figsize=(7, 3.2))
f, s = spec(naive16, 16000); ax.plot(f, s, label="naive (no low-pass): tone folds to 6 kHz", color="C3", lw=1)
f, s = spec(good, 16000);    ax.plot(f, s, label="load_audio resampler: tone removed (below -150 dB, off-scale)", color="C2", lw=1)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("level (dB)"); ax.set_ylim(-150, 10)
ax.set_title("Why the resampler must low-pass first (10 kHz tone, 44.1 kHz -> 16 kHz)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_aliasing.png")); plt.close(fig)

# ---- Figure 3: normalisation makes loudness consistent
fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for n, lab in [("speech_like_quiet_16k.wav", "quiet (-40 dBFS)"), ("speech_like_loud_clipped_16k.wav", "loud / clipped")]:
    r, _ = sf.read(os.path.join(S, n)); ax[0].plot(r[:16000], lw=.4, label=lab)
    o, _, _ = load_audio_with_report(os.path.join(S, n)); ax[1].plot(o[:16000], lw=.4, label=lab)
ax[0].set_title("Before"); ax[1].set_title("After peak-normalisation (0.9)"); ax[0].legend(fontsize=7, loc="upper right")
ax[0].set_ylim(-1, 1); fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_normalisation.png")); plt.close(fig)

# ---- Results table for every sample file
rows = []
for n in sorted(os.listdir(S)):
    try:
        y, sr, r = load_audio_with_report(os.path.join(S, n))
        rows.append([n, r.decoder, r.original_sr, r.original_channels, f"{r.original_duration_s:.2f}",
                     f"{r.peak_before:.3f}", f"{r.dc_offset_removed:+.3f}", f"{r.gain_db:+.1f}",
                     f"{r.peak_after:.3f}", len(y), "; ".join(r.warnings) or "-"])
    except Exception as e:
        rows.append([n, "-", "-", "-", "-", "-", "-", "-", "-", "-", f"ERROR {e}"])
hdr = ["file", "decoder", "in_sr", "in_ch", "in_s", "peak_before", "dc", "gain_dB", "peak_after", "out_samples", "warnings"]
with open(os.path.join(ROOT, "docs", "results_table.csv"), "w", newline="") as fh:
    csv.writer(fh).writerow(hdr); csv.writer(fh).writerows(rows)
for r in rows: print(" | ".join(map(str, r)))
