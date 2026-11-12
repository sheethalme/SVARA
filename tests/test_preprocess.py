import io, os, shutil, sys
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from synth import speech_like, tone  # noqa: E402

import preprocess as pp  # noqa: E402
from audio_io import AudioLoadError, load_audio  # noqa: E402
from preprocess import (PreprocessConfig, bandpass_filter, bandpass_response, clean_audio,  # noqa: E402
                        clean_audio_with_stages, detect_speech, frame_energy_db, pre_emphasis,
                        pre_emphasis_response, spectral_subtraction, trim_silence)

SR = 16000
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "sample_audio")


def gain_at(f, filt, dur=1.0):
    x = tone(f, dur, SR, 0.5)
    y = filt(x)
    mid = slice(int(0.2 * SR), -int(0.2 * SR))
    return np.max(np.abs(y[mid])) / 0.5


def make_clip(snr_db=20.0, lead=1.0, trail=1.0, seed=0, kind="white"):
    rng = np.random.default_rng(seed)
    sp = speech_like(2.5, SR, seed=1)
    clean = np.concatenate([np.zeros(int(lead * SR)), sp, np.zeros(int(trail * SR))]).astype(np.float64)
    n = len(clean)
    if kind == "white":
        noise = rng.standard_normal(n)
    else:
        X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X)); f[0] = 1
        noise = np.fft.irfft(X / np.sqrt(f), n)
    noise *= np.sqrt(np.mean(sp.astype(float) ** 2) / 10 ** (snr_db / 10)) / np.std(noise)
    return clean.astype(np.float32), (clean + noise).astype(np.float32), lead, lead + 2.5


def sisnr(ref, est):
    g = np.dot(est, ref) / np.dot(ref, ref)
    return 10 * np.log10(np.sum((g * ref) ** 2) / np.sum((est - g * ref) ** 2))


# ------------------------------------------------------------------ pre-emphasis
def test_pre_emphasis_formula_exact():
    x = np.array([1.0, 2.0, 4.0, 3.0], np.float32)
    np.testing.assert_allclose(pre_emphasis(x, 0.97), [1.0, 2 - .97, 4 - 2 * .97, 3 - 4 * .97], atol=1e-6)


def test_pre_emphasis_gain_rises_with_frequency():
    g = [gain_at(f, lambda x: pre_emphasis(x, 0.97)) for f in (100, 500, 1000, 4000, 7000)]
    assert g == sorted(g)
    assert g[0] < 0.25 and g[-1] > 1.8          # low cut, high boosted (~+5.9 dB at 8 kHz)


def test_pre_emphasis_response_matches_theory():
    w, db = pre_emphasis_response(0.97, SR)
    expected = 20 * np.log10(np.sqrt(1 + 0.97 ** 2 - 2 * 0.97 * np.cos(2 * np.pi * w / SR)))
    np.testing.assert_allclose(db, expected, atol=1e-6)
    assert db[-1] == pytest.approx(5.9, abs=0.1)


def test_pre_emphasis_flattens_spectral_tilt():
    # a signal with ~ -6 dB/octave tilt (brown-ish) becomes flatter
    rng = np.random.default_rng(0)
    X = np.fft.rfft(rng.standard_normal(SR * 4)); f = np.arange(len(X)); f[0] = 1
    x = np.fft.irfft(X / f, SR * 4).astype(np.float32)
    def tilt(sig):
        P = np.abs(np.fft.rfft(sig)) ** 2; fr = np.fft.rfftfreq(len(sig), 1 / SR)
        lo = P[(fr > 200) & (fr < 400)].mean(); hi = P[(fr > 4000) & (fr < 6000)].mean()
        return 10 * np.log10(lo / hi)
    assert tilt(pre_emphasis(x)) < tilt(x) - 15


def test_pre_emphasis_empty_and_dtype():
    assert pre_emphasis(np.zeros(0, np.float32)).size == 0
    assert pre_emphasis(np.ones(5)).dtype == np.float32


# ------------------------------------------------------------------ band-pass
def test_bandpass_passband_and_stopband():
    f = lambda x: bandpass_filter(x, SR)
    for hz in (500, 1000, 2000):                       # well inside 300-3400: untouched
        assert gain_at(hz, f) > 0.95, hz
    assert gain_at(3000, f) > 0.75                     # near the upper edge: <= ~2 dB loss
    assert gain_at(100, f) < 0.05                      # hum / rumble
    assert gain_at(50, f) < 0.001
    assert gain_at(6000, f) < 0.05                     # hiss
    assert gain_at(7500, f) < 0.01


def test_bandpass_edges_are_minus_6db_zero_phase():
    for hz in (300, 3400):
        assert gain_at(hz, lambda x: bandpass_filter(x, SR)) == pytest.approx(0.5, abs=0.03)


def test_bandpass_is_zero_phase_no_delay():
    x = tone(1000, 1.0, SR, 0.5)
    y = bandpass_filter(x, SR)
    mid = slice(4000, -4000)
    assert np.dot(x[mid], y[mid]) / (np.linalg.norm(x[mid]) * np.linalg.norm(y[mid])) > 0.999


def test_bandpass_response_agrees_with_measurement():
    w, db = bandpass_response(SR)
    k = np.argmin(np.abs(w - 100)); assert db[k] < -30
    k = np.argmin(np.abs(w - 1000)); assert abs(db[k]) < 0.5
    k = np.argmin(np.abs(w - 300)); assert db[k] == pytest.approx(-6.0, abs=0.3)


def test_bandpass_very_short_signal_ok():
    y = bandpass_filter(np.ones(10, np.float32), SR)
    assert y.shape == (10,) and np.all(np.isfinite(y))


def test_bandpass_upper_edge_clamped_below_nyquist():
    y = bandpass_filter(tone(1000, 0.5, SR), SR, 300, 9000)   # 9 kHz > Nyquist: clamped
    assert np.all(np.isfinite(y))


# ------------------------------------------------------------------ noise reduction
@pytest.mark.parametrize("kind,snr", [("white", 20), ("white", 10), ("pink", 20), ("pink", 10), ("pink", 5)])
def test_spectral_subtraction_improves_snr(kind, snr):
    clean, noisy, *_ = make_clip(snr, kind=kind)
    ref = clean_audio_with_stages(clean, SR, denoise=None, vad=False, normalize_output=None).signal
    a = clean_audio_with_stages(noisy, SR, denoise=None, vad=False, normalize_output=None).signal
    b = clean_audio_with_stages(noisy, SR, vad=False, normalize_output=None).signal
    assert sisnr(ref, b) > sisnr(ref, a) + 3.0


def test_spectral_subtraction_lowers_noise_floor():
    _, noisy, *_ = make_clip(10)
    rep = clean_audio_with_stages(noisy, SR).report
    assert rep.noise_reduction_applied
    assert rep.noise_floor_after_db < rep.noise_floor_before_db - 6


def test_spectral_subtraction_preserves_length_and_finite():
    _, noisy, *_ = make_clip(10)
    y, info = spectral_subtraction(noisy, SR)
    assert y.shape == noisy.shape and y.dtype == np.float32 and np.all(np.isfinite(y))
    assert info["applied"]


def test_spectral_subtraction_keeps_speech_energy_in_clean_input():
    clean, *_ = make_clip(60)       # almost noise-free clip with pauses
    y, info = spectral_subtraction(clean, SR)
    sp = slice(SR + 4000, 3 * SR)
    ratio = np.sqrt(np.mean(y[sp] ** 2) / np.mean(clean[sp] ** 2))
    assert 0.85 < ratio < 1.05


def test_spectral_subtraction_skipped_without_pauses():
    y, info = spectral_subtraction(speech_like(2.0, SR)[:0] if False else tone(500, 2.0, SR), SR)
    assert not info["applied"] and "pauses" in info["reason"]
    np.testing.assert_array_equal(y, tone(500, 2.0, SR))


def test_spectral_subtraction_short_clip_skipped():
    y, info = spectral_subtraction(np.ones(100, np.float32), SR)
    assert not info["applied"]


def test_noisereduce_backend():
    pytest.importorskip("noisereduce")
    clean, noisy, *_ = make_clip(10)
    ref = clean_audio_with_stages(clean, SR, denoise=None, vad=False, normalize_output=None).signal
    a = clean_audio_with_stages(noisy, SR, denoise=None, vad=False, normalize_output=None).signal
    b = clean_audio_with_stages(noisy, SR, denoise="noisereduce", vad=False, normalize_output=None).signal
    assert b.shape == a.shape and sisnr(ref, b) > sisnr(ref, a) + 3


def test_noisereduce_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "noisereduce", None)
    with pytest.raises(ImportError, match="noisereduce"):
        clean_audio(make_clip(10)[1], SR, denoise="noisereduce")


# ------------------------------------------------------------------ VAD
@pytest.mark.parametrize("snr", [30, 20, 10, 5])
def test_vad_finds_speech_boundaries(snr):
    _, noisy, t0, t1 = make_clip(snr)
    res = clean_audio_with_stages(noisy, SR)
    assert res.vad.reliable and res.vad.speech_detected
    start, end = res.vad.segments[0][0] / SR, res.vad.segments[-1][1] / SR
    assert abs(start - t0) < 0.1 and abs(end - t1) < 0.1


def test_vad_trims_silence_but_keeps_pad():
    _, noisy, t0, t1 = make_clip(20)
    res = clean_audio_with_stages(noisy, SR)
    assert res.report.output_duration_s == pytest.approx((t1 - t0) + 0.3, abs=0.15)
    assert res.report.trimmed_start_s == pytest.approx(t0 - 0.15, abs=0.1)
    assert res.report.trimmed_end_s == pytest.approx(1.0 - 0.15, abs=0.1)


def test_vad_keeps_internal_pauses_and_bridges_short_gaps():
    sp = speech_like(1.5, SR, seed=1)
    gap = np.zeros(int(0.25 * SR), np.float32)               # 250 ms pause < 300 ms
    x = np.concatenate([np.zeros(SR), sp, gap, sp, np.zeros(SR)])
    x = x + 0.003 * np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    v = detect_speech(x, SR)
    assert len(v.segments) == 1


def test_vad_splits_on_long_pause_but_trim_keeps_inside():
    sp = speech_like(1.0, SR, seed=1)
    x = np.concatenate([np.zeros(SR), sp, np.zeros(int(0.8 * SR)), sp, np.zeros(SR)])
    x = x + 0.003 * np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    v = detect_speech(x, SR)
    assert len(v.segments) == 2
    trimmed, s, e = trim_silence(x, v, SR, 150)
    assert (e - s) / SR == pytest.approx(1 + 0.8 + 1 + 0.3, abs=0.15)   # inner pause intact


def test_vad_ignores_short_click():
    x = 0.002 * np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)
    x[SR: SR + 400] += 0.8 * np.hanning(400).astype(np.float32)   # 25 ms click
    sp = speech_like(1.0, SR, seed=1)
    x[int(1.8 * SR): int(2.8 * SR)] += sp
    v = detect_speech(x, SR)
    assert len(v.segments) == 1 and v.segments[0][0] / SR > 1.5


def test_vad_speech_at_very_start_and_end_clamped():
    sp = speech_like(2.0, SR, seed=1)
    x = np.concatenate([sp, np.zeros(SR)]) + 0.002 * np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)
    res = clean_audio_with_stages(x.astype(np.float32), SR, denoise=None)
    assert res.trim_start == 0 and 0 < res.trim_end <= len(x)


def test_vad_pure_tone_unreliable_and_untrimmed():
    res = clean_audio_with_stages(tone(500, 2.0, SR), SR)
    assert not res.vad.reliable and not res.report.speech_detected
    assert res.report.output_duration_s == pytest.approx(2.0, abs=0.01)
    assert any("untrimmed" in w for w in res.report.warnings)


def test_vad_threshold_adapts_to_level():
    _, noisy, *_ = make_clip(15)
    a = detect_speech(noisy, SR); b = detect_speech(noisy * 0.1, SR)
    assert b.threshold_db == pytest.approx(a.threshold_db - 20, abs=0.5)
    assert a.segments == b.segments


def test_frame_energy_db_known_values():
    x = np.full(1600, 0.1, np.float32)
    e = frame_energy_db(x, 400, 160)
    assert e[0] == pytest.approx(-20.0, abs=0.01)


# ------------------------------------------------------------------ full pipeline
def test_clean_audio_contract():
    _, noisy, *_ = make_clip(10)
    y = clean_audio(noisy, SR)
    assert y.dtype == np.float32 and y.ndim == 1
    assert np.max(np.abs(y)) == pytest.approx(0.9, abs=1e-3) and np.all(np.isfinite(y))


def test_clean_audio_accepts_all_1a_input_types(tmp_path):
    clean, noisy, *_ = make_clip(15)
    p = tmp_path / "a.wav"; sf.write(p, noisy, SR)
    b = io.BytesIO(); sf.write(b, noisy, SR, format="WAV")
    sig, _ = load_audio(noisy, SR)
    ref = clean_audio(noisy, SR)
    for src in (str(p), p, b.getvalue(), io.BytesIO(b.getvalue()), (noisy, SR), sig):
        out = clean_audio(src)
        assert abs(len(out) - len(ref)) <= 2


def test_clean_audio_handles_44k_stereo_file():
    out = clean_audio(os.path.join(SAMPLES, "speech_like_48k_stereo.wav"))
    assert out.ndim == 1 and np.all(np.isfinite(out))


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not installed (needed for WebM/M4A only)")


@pytest.mark.parametrize("name", [
    "speech_like.mp3",
    pytest.param("speech_like.webm", marks=needs_ffmpeg),
    pytest.param("speech_like.m4a", marks=needs_ffmpeg),
])
def test_clean_audio_compressed_inputs(name):
    out = clean_audio(os.path.join(SAMPLES, name))
    assert out.ndim == 1 and len(out) > 8000


def test_silence_input_is_safe():
    res = clean_audio_with_stages(np.zeros(SR, np.float32), SR)
    assert np.all(res.signal == 0) and not res.report.speech_detected
    assert np.all(np.isfinite(res.signal))


def test_very_short_clip_is_safe():
    out = clean_audio(speech_like(1.0, SR)[:1000], SR)        # 62 ms
    assert out.ndim == 1 and np.all(np.isfinite(out))


def test_stage_switches_for_ablation():
    _, noisy, *_ = make_clip(10)
    full = clean_audio_with_stages(noisy, SR)
    none = clean_audio_with_stages(noisy, SR, pre_emphasis=None, bandpass=False, denoise=None,
                                   vad=False, normalize_output=None)
    assert none.report.stages_applied == []
    assert len(none.signal) == len(noisy)
    assert [s.split("(")[0] for s in full.report.stages_applied] == [
        "pre_emphasis", "bandpass", "noise_reduction", "vad_trim", "normalize"]
    only_bp = clean_audio_with_stages(noisy, SR, pre_emphasis=None, denoise=None, vad=False)
    assert [s.split("(")[0] for s in only_bp.report.stages_applied] == ["bandpass", "normalize"]


def test_stages_are_time_aligned_and_consistent():
    _, noisy, *_ = make_clip(10)
    r = clean_audio_with_stages(noisy, SR)
    n = len(r.stages["input"])
    for k in ("pre_emphasis", "bandpass", "denoised"):
        assert len(r.stages[k]) == n
    assert len(r.stages["trimmed"]) == r.trim_end - r.trim_start
    assert r.vad is not None and r.vad.mask.dtype == bool


def test_idempotent_enough_on_second_pass():
    _, noisy, *_ = make_clip(10)
    once = clean_audio(noisy, SR)
    twice = clean_audio(once, SR)
    assert abs(len(twice) - len(once)) < 0.4 * SR          # does not keep eating the speech
    assert np.all(np.isfinite(twice))


def test_output_range_always_valid():
    for snr in (30, 5, 0):
        y = clean_audio(make_clip(snr, kind="pink")[1], SR)
        assert np.max(np.abs(y)) <= 1.0


def test_config_validation():
    for bad in (dict(pre_emphasis=1.5), dict(low_hz=4000, high_hz=3000), dict(low_hz=9000, high_hz=9500),
                dict(filter_order=0), dict(denoise="magic"), dict(ss_hop=9999), dict(ss_gain_floor=1.0),
                dict(vad_threshold_fraction=0), dict(vad_hop_ms=50, vad_frame_ms=25)):
        with pytest.raises(ValueError):
            clean_audio(make_clip(10)[1], SR, **bad)
    with pytest.raises(TypeError):
        clean_audio(make_clip(10)[1], SR, not_a_setting=1)


def test_1a_errors_propagate():
    with pytest.raises(AudioLoadError):
        clean_audio(b"not audio")
    with pytest.raises(AudioLoadError):
        clean_audio("/no/such/file.wav")


def test_bare_array_assumed_16k():
    x = make_clip(10)[1]
    np.testing.assert_allclose(clean_audio(x), clean_audio(x, 16000))


def test_report_serialisable():
    r = clean_audio_with_stages(make_clip(10)[1], SR).report.to_dict()
    import json; json.dumps(r, default=str)
    assert r["audio_input"]["output_sr"] == 16000
