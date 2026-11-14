"""Try SVARA 1A + 1B + 2A + 2B with your own microphone.      Run:  streamlit run scripts/try_it_app.py

Record (or upload) an answer, then compare the audio before and after the preprocessing chain.
The switches in the sidebar turn each stage on/off, so you can hear what every stage does.
"""
import os, sys
os.environ.setdefault("SVARA_WHISPER_MODEL", "large-v3-turbo")   # much more accurate than "small"; override via env
import numpy as np
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import AudioLoadError, load_audio_with_report
import preprocess as pp
import stt
import extractor
st.set_page_config(page_title="SVARA 1A+1B+2A+2B test", layout="wide")
st.title("SVARA: audio input (1A) + preprocessing (1B) + speech-to-text (2A) + extraction (2B)")
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
# ----------------------------------------------------------------------------------- 2A
st.divider()
st.subheader("2A: speech-to-text and language ID")
LANGS = {"auto-detect": None, "ml - Malayalam": "ml", "kn - Kannada": "kn", "hi - Hindi": "hi", "en - English": "en"}
ENGINES = {"auto (cloud first, local fallback)": "auto", "local Whisper only": "local", "cloud only": "cloud"}
q1, q2, q3 = st.columns(3)
lang_label = q1.selectbox("Spoken language", list(LANGS), index=list(LANGS).index("en - English"))
eng_label = q2.selectbox("Engine", list(ENGINES))
which = q3.radio("Audio given to the recogniser", ["after 1B (cleaned)", "after 1A only (raw)"], horizontal=True)
d = stt.get_default()
st.caption(f"Cloud engine: {'READY (' + d.cloud.model_name + ')' if d.cloud.available else 'not configured (set GROQ_API_KEY or OPENAI_API_KEY)'}"
           f"  |  Local engine: {'faster-whisper installed, model ' + d.cfg.local_model if d.local.available else 'faster-whisper NOT installed'}"
           "  |  the first local transcription loads the model (about 10 s).")
_akey = (name, len(x), round(float(np.sum(np.abs(x))), 3), which, lang_label, eng_label)
_go = st.button("Re-transcribe") or st.session_state.get("stt_key") != _akey
if _go:
    st.session_state["stt_key"] = _akey
    try:
        with st.spinner("Transcribing..."):
            st.session_state["stt"] = stt.transcribe_full(
                res.signal if which.startswith("after 1B") else x, language=LANGS[lang_label], engine=ENGINES[eng_label])
    except (stt.STTError, AudioLoadError, ValueError) as e:
        st.session_state["stt"] = None; st.error(str(e))
r3 = st.session_state.get("stt")
if r3 is not None:
    (st.success if r3.reliable else st.warning)("Reliable result" if r3.reliable else "Please check this result (see warnings)")
    st.markdown(f"### {r3.text or '(nothing recognised)'}")
    k = st.columns(4)
    k[0].metric("Language", stt.SUPPORTED_LANGUAGES.get(r3.language, r3.language),
                f"{r3.language_confidence:.0%} confidence" if r3.language_confidence is not None else None)
    k[1].metric("Engine", r3.engine or "-", "fallback" if r3.fallback_used else None)
    k[2].metric("Model", r3.model or "-")
    k[3].metric("Time", f"{r3.latency_s:.1f} s")
    if r3.fallback_reason:
        st.info(f"Cloud was not used because: {r3.fallback_reason}")
    for w in r3.warnings:
        st.warning(w)
    with st.expander("Full 2A result"):
        st.json(r3.to_dict())

# ----------------------------------------------------------------------------------- 2B
st.divider()
st.subheader("2B: field extraction and confirmation")
_fields = extractor.list_fields()
_prev = st.session_state.get("stt")                       # the 2A result, if you pressed Transcribe above
_lang = _prev.language if _prev is not None and _prev.language in ("ml", "kn", "hi", "en") else None
_default = _prev.text if _prev is not None and _prev.text else ""
_llm = extractor.get_extractor().llm
st.caption("LLM helper: " + ("ON (" + getattr(_llm, "model_name", "custom") + ")" if _llm is not None else
           "not configured, rules only (set GROQ_API_KEY or OPENAI_API_KEY to enable)") +
           "  |  phone, Aadhaar, PIN and e-mail are never sent to an LLM.")
b1, b2 = st.columns([1, 3])
_field = b1.selectbox("Form field", _fields, index=_fields.index("age") if "age" in _fields else 0, key="field_2b")
_answer = b2.text_area("What the user said (filled from 2A, or type a test sentence)", value=_default,
                       key="answer_2b_" + str(abs(hash(_default))), height=80)
_ekey = (_answer, _field)
if (st.button("Extract field", type="primary", key="btn_2b") or
        (_answer.strip() and st.session_state.get("ext_key") != _ekey)):
    st.session_state["ext_key"] = _ekey
    st.session_state["ext_2b"] = extractor.extract_field_full(_answer, _field, language=_lang)
import re as _re
_CUES = [  # (field, words that signal it); first match wins, so the more specific ones come first
    ("email", r"e-?mail|@|\bat the rate\b"), ("date of birth", r"\bborn\b|birth|\bdob\b"),
    ("age", r"years? old|\bage\b|\byears?\b"), ("phone", r"phone|mobile|contact|\bnumber\b"),
    ("pin code", r"\bpin\b|pin ?code|postal"), ("aadhaar number", r"aadhaar|aadhar"),
    ("blood group", r"blood"), ("gender", r"\b(male|female|man|woman|gender)\b"),
    ("address", r"\bstay|\blive|reside|address|\bfrom\b|\bat\b"), ("name", r"\bname\b|this is|\bi am\b|\bi'm\b"),
]
def _all_fields(text):
    """Split one long answer into clauses and extract the field each clause is about."""
    out, seen = [], set()
    for clause in _re.split(r"[,;]|\s+and\s+|(?<=[a-z0-9]{2})[.!?]\s+", text):
        clause = clause.strip(" .")
        low = clause.lower()
        for fld, pat in _CUES:
            if fld in _fields and fld not in seen and _re.search(pat, low):
                r = extractor.extract_field_full(clause, fld, language=_lang)
                if r.value is not None:
                    seen.add(fld); out.append((fld, r))
                    break
    return out
if _answer.strip():
    _found = _all_fields(_answer)
    _form = st.session_state.setdefault("form", {})
    if st.session_state.get("form_src") != _answer:           # new sentence: fill the form automatically
        st.session_state["form_src"] = _answer
        for f, r in _found:
            _form[f] = {"Value": r.display or r.value, "Status": r.status, "Confidence": f"{r.confidence:.0%}",
                        "Check?": "" if r.reliable else "please check"}
    st.markdown("#### Details (filled automatically, corrections update this table)")
    if _form:
        st.table([{"Field": k, **v} for k, v in _form.items()])
        if st.button("Clear details"):
            st.session_state["form"] = {}; st.rerun()
    else:
        st.info("No details recognised in this sentence.")

_e = st.session_state.get("ext_2b")
if _e is not None:
    (st.success if _e.reliable else st.warning)("Reliable result" if _e.reliable else "Please check this value (see issues)")
    st.markdown(f"### {_e.display or _e.value or '(nothing found)'}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stored value", _e.value if _e.value is not None else "-")
    c2.metric("Status", _e.status)
    c3.metric("Confidence", f"{_e.confidence:.0%}")
    c4.metric("Method", _e.method)
    for _i in _e.issues:
        st.warning(_i)
    if _e.llm_error:
        st.info("LLM not used: " + _e.llm_error)
    if _e.value is not None:
        st.caption("Read back as: " + extractor.spoken_form(_e.field, _e.value, _lang or "en"))
    if st.session_state.get("ext_msg"):
        st.success(st.session_state.pop("ext_msg"))
    _n = st.session_state.get("reply_n", 0)
    _reply = st.text_input("Type the user's reply to 'Is that correct?'  (try: yes / no, say again / no, my age is 26)",
                           key=f"reply_2b_{_n}")
    if _reply:
        _c = extractor.interpret_confirmation_full(_reply, _lang)
        _o = extractor.handle_confirmation(_reply, _e.field, _lang)
        st.write(f"**Reply understood as:** `{_c.label}`" + (f" (reason: {_c.reason})" if _c.reason else "") +
                 f"   |   **Next step:** `{_o.action}`" + (f"   |   **New value:** `{_o.value}`" if _o.value else ""))
        _form = st.session_state.setdefault("form", {})
        _upd = []                                             # (field, ExtractionResult) corrections to apply
        if _o.action == "update" and _o.value:                # correction for the field being confirmed
            _upd.append((_e.field, _o.extraction or extractor.extract_field_full(_o.value, _e.field, language=_lang)))
        elif _c.label == "no":                                # correction about ANOTHER field ("no my name is ...")
            _upd = _all_fields(_c.correction or _reply)
        if _upd:
            for f, r in _upd:
                _form[f] = {"Value": r.display or r.value, "Status": r.status, "Confidence": f"{r.confidence:.0%}",
                            "Check?": "" if r.reliable else "please check"}
                if f == _e.field:
                    st.session_state["ext_2b"] = r
            st.session_state["ext_msg"] = "Updated: " + ", ".join(f"{f} = {r.display or r.value}" for f, r in _upd)
            st.session_state["reply_n"] = _n + 1              # clear the reply box
            st.rerun()
        elif _o.action == "accept" and _e.value is not None:
            _form[_e.field] = {"Value": _e.display or _e.value, "Status": "confirmed", "Confidence": f"{_e.confidence:.0%}", "Check?": ""}
    with st.expander("Full 2B result"):
        st.json(_e.to_dict())


with st.expander("Full reports (1A and 1B)"):
    st.json({"1A": r1.to_dict(), "1B": r2.to_dict()})

