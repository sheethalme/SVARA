"""Try SVARA 1A + 1B + 1C with your own microphone.      Run:  streamlit run scripts/try_it_app.py

Record (or upload) an answer, then compare the audio before and after the preprocessing chain.
The switches in the sidebar turn each stage on/off, so you can hear what every stage does;
the 1C Signal Lab shows the features (waveform, spectrogram, MFCC, pitch) before and after.
"""
import os, sys
import numpy as np
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import AudioLoadError, load_audio_with_report
import preprocess as pp
from features import extract_features, summarize_features
from plots import plot_before_after, plot_feature_overview

st.set_page_config(page_title="SVARA 1A+1B+1C test", layout="wide")
st.title("SVARA: audio input (1A) + preprocessing (1B) + features (1C) test")
st.caption("Tip: leave about half a second of silence before and after you speak.")

with st.sidebar:
    st.header("1B settings")
    pre = st.checkbox("Pre-emphasis (0.97)", True)
    bp = st.checkbox("Band-pass 300-3400 Hz", True)
    dn = st.selectbox("Noise reduction", ["spectral_subtraction", "noisereduce", "off"])
    vad = st.checkbox("VAD silence trimming", True)
    extra = st.slider("Add white noise to test (SNR dB, 40 = none)", 0, 40, 40)

source, name = None, ""
c1, c2 = st.columns(2)
with c1:
    mic = st.audio_input("Record your voice")
with c2:
    up = st.file_uploader("...or upload audio", type=["wav", "mp3", "flac", "ogg", "m4a", "webm"])
    if st.button("Use built-in noisy test clip"):
        st.session_state["builtin"] = True
if mic is not None:
    source, name = mic, "microphone recording"
elif up is not None:
    source, name = up, up.name
elif st.session_state.get("builtin"):
    source, name = os.path.join(ROOT, "sample_audio", "noisy", "white_10dB.wav"), "built-in noisy test clip"

if source is None:
    st.info("Record, upload, or use the built-in clip to start.")
    st.stop()

try:
    x, sr, r1 = load_audio_with_report(source)
except AudioLoadError as e:
    st.error(f"Could not load the audio: {e}"); st.stop()
if extra < 40:
    loud = x[np.abs(x) > 0.1 * np.max(np.abs(x))]
    nz = np.random.default_rng(0).standard_normal(len(x))
    nz *= np.sqrt(np.mean(loud.astype(float) ** 2) / 10 ** (extra / 10)) / np.std(nz)
    x = (x + nz).astype(np.float32); x = x / np.max(np.abs(x)) * 0.9

res = pp.clean_audio_with_stages(x, sr, pre_emphasis=0.97 if pre else None, bandpass=bp,
                                 denoise=None if dn == "off" else dn, vad=vad)
r2 = res.report
st.subheader(f"Result for: {name}")
for w in r1.warnings + r2.warnings:
    st.warning(w)
m = st.columns(4)
m[0].metric("Input", f"{r1.original_sr} Hz / {r1.original_channels} ch")
m[1].metric("Duration", f"{r2.input_duration_s:.2f} s -> {r2.output_duration_s:.2f} s")
m[2].metric("Noise floor", f"{r2.noise_floor_after_db:.0f} dB", f"{r2.noise_floor_after_db - r2.noise_floor_before_db:+.1f} dB")
m[3].metric("Speech found", "yes" if r2.speech_detected else "no")
a, b = st.columns(2)
a.write("**Before** (after 1A only)"); a.audio(x, sample_rate=sr)
b.write("**After** (1A + 1B)"); b.audio(res.signal, sample_rate=res.sr)

fig, ax = plt.subplots(2, 2, figsize=(11, 5))
off = res.trim_start / sr
for j, (sig, o, ttl) in enumerate([(x, 0.0, "Before"), (res.signal, off, "After")]):
    ax[0, j].plot(np.arange(len(sig)) / sr + o, sig, lw=.3); ax[0, j].set_title(ttl); ax[0, j].set_xlim(0, len(x) / sr)
    f, t, P = sps.spectrogram(sig, sr, nperseg=512, noverlap=384)
    ax[1, j].pcolormesh(t + o, f, 10 * np.log10(P + 1e-12), vmin=-100, vmax=-20, shading="auto", cmap="magma")
    ax[1, j].set_xlim(0, len(x) / sr); ax[1, j].set_ylabel("Hz")
fig.tight_layout(); st.pyplot(fig)
if res.vad is not None:
    v = res.vad; f2, ax2 = plt.subplots(figsize=(11, 2.4))
    ax2.plot(v.frame_times_s, v.energy_db, lw=.9); ax2.axhline(v.threshold_db, color="r", ls="--")
    ax2.fill_between(v.frame_times_s, v.energy_db.min(), v.energy_db.max(), where=v.mask, color="g", alpha=.2)
    ax2.set_title("VAD: frame energy (dB), threshold (red), speech (green)"); st.pyplot(f2)

st.subheader("1C Signal Lab: features before vs after preprocessing")
fb, fa = extract_features(x, sr), extract_features(res.signal, res.sr)
st.pyplot(plot_before_after(res, feats_before=fb, feats_after=fa))
sb, sa = summarize_features(fb), summarize_features(fa)
st.table({"feature": list(sb), "before": [f"{v:.3f}" for v in sb.values()],
          "after": [f"{v:.3f}" for v in sa.values()]})
st.caption("Pitch is measured more reliably on the 'before' signal: the 300 Hz band-pass removes "
           "the fundamental of low voices, so fewer frames are classed as voiced after 1B.")
with st.expander("All features of the cleaned signal (mel spectrogram, energy, ZCR, centroid)"):
    st.pyplot(plot_feature_overview(res.signal, res.sr, fa, title="Cleaned signal"))
plt.close("all")
with st.expander("Full reports (1A and 1B)"):
    st.json({"1A": r1.to_dict(), "1B": r2.to_dict()})
