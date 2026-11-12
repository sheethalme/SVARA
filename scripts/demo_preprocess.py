"""Figures + measured results for 1B.  Run: python scripts/demo_preprocess.py"""
import csv, os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
import preprocess as pp
from make_noisy_samples import make_clip
SR = 16000
F = os.path.join(ROOT, "docs", "figures"); os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"blue": "#2a6fbb", "orange": "#d9822b", "green": "#2e8b57", "red": "#c0392b", "grey": "#777777"}

def sisnr(ref, est):
    g = np.dot(est, ref) / np.dot(ref, ref)
    return 10 * np.log10(np.sum((g * ref) ** 2) / np.sum((est - g * ref) ** 2))

# ---- Fig 4: pre-emphasis response + effect on a spectrum
w, db = pp.pre_emphasis_response(0.97, SR)
clean, noisy, t0, t1 = make_clip(20)
x = clean[int(1.2 * SR):int(3.2 * SR)]; y = pp.pre_emphasis(x)
def welch_db(s): f, p = sps.welch(s, SR, nperseg=1024); return f, 10 * np.log10(p + 1e-14)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
ax[0].plot(w, db, color=C["blue"]); ax[0].axhline(0, color=C["grey"], lw=.6)
ax[0].set(title="Pre-emphasis filter gain, alpha = 0.97", xlabel="frequency (Hz)", ylabel="gain (dB)")
f, a = welch_db(x); f, b = welch_db(y)
ax[1].plot(f, a, color=C["grey"], label="before"); ax[1].plot(f, b, color=C["orange"], label="after")
ax[1].set(title="Effect on a speech-like spectrum", xlabel="frequency (Hz)", ylabel="power (dB/Hz)"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(F, "fig4_preemphasis.png")); plt.close(fig)

# ---- Fig 5: band-pass response
w, db = pp.bandpass_response(SR)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(w, np.maximum(db, -80), color=C["blue"]); ax.axvspan(300, 3400, color=C["green"], alpha=.12, label="pass band 300-3400 Hz")
ax.axhline(-6, color=C["grey"], lw=.6, ls="--"); ax.text(7000, -4.5, "-6 dB", color=C["grey"], ha="right")
ax.set(title="Band-pass filter: Butterworth order 4, forward-backward (zero phase)", xlabel="frequency (Hz)", ylabel="gain (dB)", ylim=(-80, 5))
ax.legend(loc="lower center"); fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_bandpass.png")); plt.close(fig)

# ---- Fig 6: all stages, waveform + spectrogram (10 dB white noise)
clean, noisy, t0, t1 = make_clip(10)
r = pp.clean_audio_with_stages(noisy, SR)
names = [("input", "Input (10 dB white noise)"), ("bandpass", "After pre-emphasis + band-pass"),
         ("denoised", "After noise reduction"), ("trimmed", "After VAD trimming")]
fig, ax = plt.subplots(2, 4, figsize=(11, 4.6))
for j, (k, title) in enumerate(names):
    s = r.stages[k]; t = np.arange(len(s)) / SR + (r.trim_start / SR if k == "trimmed" else 0)
    ax[0, j].plot(t, s, lw=.3, color=C["blue"]); ax[0, j].set_title(title, fontsize=8); ax[0, j].set_xlim(0, len(r.stages["input"]) / SR)
    ax[0, j].set_ylim(-1.2, 1.2) if k == "input" else None
    f, tt, S = sps.spectrogram(s, SR, nperseg=512, noverlap=384)
    ax[1, j].pcolormesh(tt + (r.trim_start / SR if k == "trimmed" else 0), f, 10 * np.log10(S + 1e-12), vmin=-100, vmax=-20, shading="auto", cmap="magma")
    ax[1, j].set_xlim(0, len(r.stages["input"]) / SR); ax[1, j].set_xlabel("time (s)")
ax[0, 0].set_ylabel("amplitude"); ax[1, 0].set_ylabel("frequency (Hz)")
fig.tight_layout(); fig.savefig(os.path.join(F, "fig6_stages.png")); plt.close(fig)

# ---- Fig 7: VAD
v = r.vad
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(v.frame_times_s, v.energy_db, color=C["blue"], lw=.9, label="frame energy")
ax.axhline(v.threshold_db, color=C["red"], ls="--", lw=1, label=f"threshold {v.threshold_db:.0f} dB")
ax.axhline(v.noise_floor_db, color=C["grey"], ls=":", lw=1, label="noise floor (10th pct)")
ax.axhline(v.speech_level_db, color=C["grey"], ls=":", lw=1)
ax.fill_between(v.frame_times_s, ax.get_ylim()[0], ax.get_ylim()[1], where=v.mask, color=C["green"], alpha=.18, label="detected speech")
ax.axvline(t0, color="k", lw=.8); ax.axvline(t1, color="k", lw=.8, label="true speech start/end")
ax.axvspan(r.trim_start / SR, r.trim_end / SR, ymin=0, ymax=.04, color=C["orange"], label="kept after trim (+150 ms pad)")
ax.set(title="Energy-based voice activity detection (energy measured after noise reduction)", xlabel="time (s)", ylabel="short-time energy (dB)")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(F, "fig7_vad.png")); plt.close(fig)

# ---- Fig 8 + results: SNR improvement and VAD accuracy over 5 noise seeds
rows = []; summary = {}
for kind in ("white", "pink"):
    for snr in (20, 10, 5):
        res = {"filtered": [], "specsub": [], "noisereduce": []}; verr = []; nf = []; ok = 0
        for seed in range(5):
            clean, noisy, a, b = make_clip(snr, kind, seed=seed)
            ref = pp.clean_audio_with_stages(clean, SR, denoise=None, vad=False, normalize_output=None).signal
            run = lambda **kw: pp.clean_audio_with_stages(noisy, SR, vad=False, normalize_output=None, **kw)
            res["filtered"].append(sisnr(ref, run(denoise=None).signal))
            rs = run(); res["specsub"].append(sisnr(ref, rs.signal)); nf.append(rs.report.noise_floor_before_db - rs.report.noise_floor_after_db)
            res["noisereduce"].append(sisnr(ref, run(denoise="noisereduce").signal))
            full = pp.clean_audio_with_stages(noisy, SR)
            if full.vad.speech_detected:
                ok += 1; verr.append((abs(full.vad.segments[0][0] / SR - a) + abs(full.vad.segments[-1][1] / SR - b)) / 2)
        m = {k: (np.mean(v), np.std(v)) for k, v in res.items()}; summary[(kind, snr)] = m
        rows.append([kind, snr, *[f"{m[k][0]:.1f}" for k in ("filtered", "specsub", "noisereduce")],
                     f"{np.mean(nf):.1f}", f"{ok}/5", f"{1000 * np.mean(verr):.0f}" if verr else "-"])
hdr = ["noise", "SNR_dB", "SI-SNR filter only", "SI-SNR spectral subtraction", "SI-SNR noisereduce",
       "noise floor drop dB", "speech found", "mean VAD boundary error ms"]
with open(os.path.join(ROOT, "docs", "results_1b.csv"), "w", newline="") as fh:
    cw = csv.writer(fh); cw.writerow(hdr); cw.writerows(rows)
for rr in rows: print(" | ".join(map(str, rr)))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for j, kind in enumerate(("white", "pink")):
    xs = np.arange(3); wd = .26
    for i, (k, lab, col) in enumerate([("filtered", "filter only", C["grey"]), ("specsub", "+ spectral subtraction", C["blue"]), ("noisereduce", "+ noisereduce", C["orange"])]):
        ax[j].bar(xs + (i - 1) * wd, [summary[(kind, s)][k][0] for s in (20, 10, 5)], wd, yerr=[summary[(kind, s)][k][1] for s in (20, 10, 5)], color=col, label=lab, capsize=2)
    ax[j].set_xticks(xs); ax[j].set_xticklabels(["20 dB", "10 dB", "5 dB"]); ax[j].set_title(f"{kind} noise"); ax[j].axhline(0, color="k", lw=.5)
    ax[j].set_xlabel("input SNR")
ax[0].set_ylabel("output SI-SNR (dB)"); ax[0].legend(fontsize=7); fig.tight_layout()
fig.savefig(os.path.join(F, "fig8_snr_gain.png")); plt.close(fig)

# ---- speed
_, noisy, *_ = make_clip(10); t = time.perf_counter()
for _ in range(5): pp.clean_audio(noisy, SR)
dt = (time.perf_counter() - t) / 5
print(f"time per 4.5 s clip: {dt*1000:.0f} ms (real-time factor {dt/4.5:.3f})")
