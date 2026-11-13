"""Figures + measured numbers for 1C.  Run: python scripts/demo_features.py

Writes docs/figures/fig9_framing.png ... fig12_signal_lab.png and docs/results_1c_features.csv.
Uses a stand-in speech clip from data/test_clips_tts (python scripts/make_tts_clips.py) when it
exists, otherwise the synthetic speech-like signal.
"""
import csv, os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import plots
import preprocess as pp
from audio_io import load_audio
from evaluate import add_noise
from synth import speech_like
SR = 16000
FIG = os.path.join(ROOT, "docs", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = plots.COLORS

tts = os.path.join(ROOT, "data", "test_clips_tts", "clip07.wav")
if os.path.exists(tts):
    x, _ = load_audio(tts); label = "TTS clip: 'my date of birth is fifteenth march two thousand'"
else:
    x = speech_like(3.0, SR); label = "synthetic speech-like signal"

# ---- Fig 9: framing and windowing
f = F.extract_features(x)
i = int(np.argmax(f["energy_db"] * f["voiced"] - 1e9 * ~f["voiced"]))   # a loud voiced frame
s0 = i * 160
fr = x[s0:s0 + 400].astype(float); w = F.hamming_window(400)
fig, ax = plt.subplots(2, 2, figsize=(10, 5.4))
t = np.arange(len(x)) / SR
ax[0, 0].plot(t, x, lw=.3, color=C["grey"])
for k in range(-2, 3):
    a = (s0 + 160 * k) / SR
    ax[0, 0].axvspan(a, a + 0.025, color=C["blue"] if k == 0 else C["orange"], alpha=.25 if k == 0 else .08)
ax[0, 0].set(xlim=(s0 / SR - 0.05, s0 / SR + 0.08), title="Framing: 25 ms frames every 10 ms", xlabel="time (s)")
ax[0, 1].plot(np.arange(400) / 16, fr, color=C["grey"], lw=.8, label="frame (rectangular)")
ax[0, 1].plot(np.arange(400) / 16, fr * w, color=C["blue"], lw=.9, label="x Hamming window")
ax[0, 1].plot(np.arange(400) / 16, w * np.abs(fr).max(), color=C["red"], ls="--", lw=.8, label="window shape")
ax[0, 1].set(title="Windowing tapers the frame edges", xlabel="time in frame (ms)"); ax[0, 1].legend(fontsize=7)
freqs = np.fft.rfftfreq(512, 1 / SR)
rect = 10 * np.log10(np.abs(np.fft.rfft(fr, 512)) ** 2 / 512 + 1e-12)
ham = 10 * np.log10(F.power_spectrogram(fr[None, :], 512)[:, 0] + 1e-12)
ax[1, 0].plot(freqs, rect, color=C["grey"], lw=.8, label="rectangular")
ax[1, 0].plot(freqs, ham, color=C["blue"], lw=.8, label="Hamming")
ax[1, 0].set(title="Spectrum of the frame: less leakage with Hamming", xlabel="frequency (Hz)", ylabel="dB"); ax[1, 0].legend(fontsize=7)
for n, col in [(400, C["blue"])]:
    W = 20 * np.log10(np.abs(np.fft.rfft(F.hamming_window(n), 8192)) + 1e-12); W -= W.max()
    R = 20 * np.log10(np.abs(np.fft.rfft(np.ones(n), 8192)) + 1e-12); R -= R.max()
    fw = np.fft.rfftfreq(8192, 1 / SR)
    ax[1, 1].plot(fw, R, color=C["grey"], lw=.8, label="rectangular (side lobes -13 dB)")
    ax[1, 1].plot(fw, W, color=col, lw=.8, label="Hamming (side lobes -43 dB)")
ax[1, 1].set(xlim=(0, 500), ylim=(-90, 3), title="Window spectra (25 ms)", xlabel="frequency (Hz)", ylabel="dB"); ax[1, 1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_framing.png")); plt.close(fig)

# ---- Fig 10: MFCC computation, step by step, for one frame
fb, centres = F.mel_filterbank(SR, 512, 26)
P = F.power_spectrogram(fr[None, :], 512)[:, 0]
mel_e = fb @ P; log_mel = np.log(mel_e + 1e-10)
from scipy.fft import dct
cep = dct(log_mel, 2, norm="ortho")
fig, ax = plt.subplots(1, 4, figsize=(12, 2.9))
ax[0].plot(freqs, 10 * np.log10(P + 1e-12), color=C["blue"], lw=.8); ax[0].set(title="1. power spectrum", xlabel="Hz", ylabel="dB")
for m in range(26):
    ax[1].plot(freqs, fb[m], lw=.7, color=C["orange"] if m % 2 else C["blue"])
ax[1].set(title="2. 26 mel filters", xlabel="Hz")
ax[2].plot(centres, log_mel, "o-", ms=3, color=C["green"]); ax[2].set(title="3. log filterbank energies", xlabel="band centre (Hz)")
ax[3].bar(np.arange(26), cep, color=[C["blue"] if k < 13 else C["grey"] for k in range(26)])
ax[3].set(title="4. DCT: keep c0-c12 (blue)", xlabel="coefficient")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_mfcc_steps.png")); plt.close(fig)

# ---- Fig 11: all features of one clip
fig = plots.plot_feature_overview(x, SR, f, title=f"Waveform ({label})")
fig.savefig(os.path.join(FIG, "fig11_features.png")); plt.close(fig)

# ---- Fig 12: Signal Lab before / after at 10 dB white noise
noisy = add_noise(x, 10, "white", seed=0)
res = pp.clean_audio_with_stages(noisy, SR)
fig = plots.plot_before_after(res, titles=("Before: 10 dB white noise (1A output)", "After 1B preprocessing"))
fig.savefig(os.path.join(FIG, "fig12_signal_lab.png")); plt.close(fig)

# ---- measured numbers: what preprocessing does to the features (all TTS clips, or the synth)
clips_dir = os.path.join(ROOT, "data", "test_clips_tts")
clips = ([load_audio(os.path.join(clips_dir, n))[0] for n in sorted(os.listdir(clips_dir)) if n.endswith(".wav")]
         if os.path.isdir(clips_dir) else [speech_like(3.0, SR, seed=s) for s in range(5)])
rows = []
for snr in ("clean", 20, 10, 5):
    acc = {k: [] for k in ("centroid_in", "centroid_out", "zcr_in", "zcr_out", "voiced_in", "voiced_out",
                           "range_in", "range_out", "f0_agree")}
    for k, c in enumerate(clips):
        y = c if snr == "clean" else add_noise(c, snr, "white", seed=k)
        a = F.extract_features(load_audio(y, SR)[0]); r = pp.clean_audio_with_stages(y, SR)
        b = F.extract_features(r.signal)
        sa, sb = F.summarize_features(a), F.summarize_features(b)
        ref = F.extract_features(c)                      # pitch of the clean clip = reference
        for key, s1, s2 in [("centroid", "mean_spectral_centroid_hz", "mean_spectral_centroid_hz"), ("zcr", "mean_zcr", "mean_zcr"),
                            ("voiced", "voiced_fraction", "voiced_fraction"), ("range", "energy_range_db", "energy_range_db")]:
            acc[key + "_in"].append(sa[s1]); acc[key + "_out"].append(sb[s2])
        off = r.trim_start // 160; n = b["f0"].size
        both = b["voiced"] & ref["voiced"][off:off + n][: n] if off + n <= ref["f0"].size else np.zeros(n, bool)
        if both.any():
            acc["f0_agree"].append(np.mean(np.abs(b["f0"][both] / ref["f0"][off:off + n][both] - 1) < 0.1))
    rows.append([snr] + [f"{np.mean(acc[k]):.3f}" for k in acc])
hdr = ["SNR_dB", "centroid before Hz", "centroid after Hz", "ZCR before", "ZCR after", "voiced frac before",
       "voiced frac after", "energy range before dB", "energy range after dB", "F0 within 10% of clean (after)"]
with open(os.path.join(ROOT, "docs", "results_1c_features.csv"), "w", newline="") as fh:
    cw = csv.writer(fh); cw.writerow(hdr); cw.writerows(rows)
for r in rows: print(" | ".join(map(str, r)))

t0 = time.perf_counter()
for _ in range(5): F.extract_features(x)
dt = (time.perf_counter() - t0) / 5
print(f"extract_features on a {len(x) / SR:.1f} s clip: {dt * 1000:.0f} ms (real-time factor {dt / (len(x) / SR):.3f})")
