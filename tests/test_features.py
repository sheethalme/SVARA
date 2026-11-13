import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from synth import speech_like, tone  # noqa: E402

import features as F  # noqa: E402
import preprocess as pp  # noqa: E402
from features import FeatureConfig, extract_features, summarize_features  # noqa: E402

SR = 16000
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "sample_audio")


def harmonic(f0, dur=1.0, n=5):
    return sum(tone(f0 * k, dur, SR, 0.3 / k) for k in range(1, n + 1)).astype(np.float32)


@pytest.fixture(scope="module")
def speech():
    return speech_like(3.0, SR)


@pytest.fixture(scope="module")
def feats(speech):
    return extract_features(speech)


# ------------------------------------------------------------------ framing / windowing
def test_frame_grid_matches_1b_vad():
    x = np.random.default_rng(0).standard_normal(12345).astype(np.float32)
    f = extract_features(x)
    np.testing.assert_allclose(f["energy_db"], pp.frame_energy_db(x, 400, 160), atol=1e-6)
    assert f["frame_len"] == 400 and f["hop"] == 160
    assert f["times"][0] == pytest.approx(200 / SR) and np.allclose(np.diff(f["times"]), 0.01)


def test_frame_signal_contents_and_padding():
    x = np.arange(10, dtype=np.float32)
    fr = F.frame_signal(x, 4, 3)
    assert fr.shape == (3, 4)
    np.testing.assert_array_equal(fr[1], [3, 4, 5, 6])
    np.testing.assert_array_equal(fr[-1], [6, 7, 8, 9])
    fr = F.frame_signal(np.arange(5.0), 4, 3)           # tail zero-padded
    np.testing.assert_array_equal(fr[-1], [3, 4, 0, 0])


def test_frame_signal_long_frames_share_centres():
    x = np.arange(100, dtype=np.float64)
    short = F.frame_signal(x, 10, 5)
    long = F.frame_signal(x, 20, 5, n_frames=short.shape[0], centre_len=10)
    # centre of frame 4: short covers 20..29, long covers 15..34
    assert short[4].mean() == pytest.approx(long[4].mean())
    np.testing.assert_array_equal(long[4], np.arange(15, 35))


def test_hamming_window_formula():
    w = F.hamming_window(400)
    assert w[0] == pytest.approx(0.08) and w[-1] == pytest.approx(0.08)
    assert w.max() == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(w, np.hamming(400))
    assert F.hamming_window(1).tolist() == [1.0]


def test_windowing_reduces_leakage():
    # a tone between FFT bins: far-away bins get much less energy with the Hamming window
    x = tone(1031.25 + 15.6, 0.025, SR, 0.5)[None, :].astype(np.float64)
    rect = np.abs(np.fft.rfft(x, 512)) ** 2
    ham = F.power_spectrogram(x, 512)[:, 0] * 512
    far = slice(150, 257)                                 # > 4.6 kHz, far from the tone
    assert 10 * np.log10(rect[0, far].sum() / ham[far].sum()) > 20


# ------------------------------------------------------------------ time-domain features
def test_energy_and_zcr_known_values():
    fr = np.array([[1, -1, 1, -1], [0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0]], dtype=np.float64)
    np.testing.assert_allclose(F.short_time_energy(fr), [1.0, 0.25, 0.0])
    np.testing.assert_allclose(F.zero_crossing_rate(fr), [1.0, 0.0, 0.0])


def test_zcr_of_tone_is_2f_over_sr():
    f = extract_features(tone(1000, 1.0, SR, 0.5))
    assert np.median(f["zcr"]) == pytest.approx(2 * 1000 / SR, rel=0.03)


def test_zcr_separates_hiss_from_voiced(feats):
    noise = extract_features(0.3 * np.random.default_rng(1).standard_normal(SR).astype(np.float32))
    assert np.median(noise["zcr"]) > 0.4                  # white noise ~ 0.5
    loud = feats["energy_db"] > np.percentile(feats["energy_db"], 70)
    assert np.median(feats["zcr"][loud]) < 0.3


def test_energy_separates_speech_from_pauses(feats):
    assert np.percentile(feats["energy_db"], 95) - np.percentile(feats["energy_db"], 10) > 20


# ------------------------------------------------------------------ spectral features
@pytest.mark.parametrize("f", [500, 1500, 3000])
def test_spectral_centroid_of_tone(f):
    c = extract_features(tone(f, 0.5, SR, 0.5))["spectral_centroid"]
    assert np.median(c) == pytest.approx(f, rel=0.08)


def test_spectral_centroid_silence_is_zero():
    assert np.all(extract_features(np.zeros(SR, np.float32))["spectral_centroid"] == 0)


def test_centroid_rises_after_preemphasis(speech):
    a = extract_features(speech)["spectral_centroid"].mean()
    b = extract_features(speech, pre_emphasis=0.97)["spectral_centroid"].mean()
    assert b > a + 200


def test_mel_scale_round_trip_and_anchor():
    assert F.hz_to_mel(1000) == pytest.approx(1000, abs=0.5)    # 1 kHz ~ 1000 mel
    f = np.array([0, 300, 3400, 8000.0])
    np.testing.assert_allclose(F.mel_to_hz(F.hz_to_mel(f)), f, atol=1e-6)


def test_mel_filterbank_shape_and_triangles():
    fb, centres = F.mel_filterbank(SR, 512, 26)
    assert fb.shape == (26, 257) and centres.shape == (26,)
    assert np.all(np.diff(centres) > 0)
    assert np.all(np.diff(F.hz_to_mel(centres)) == pytest.approx(np.diff(F.hz_to_mel(centres))[0]))
    assert fb.max() <= 1.0 and np.all(fb >= 0)
    widths = (fb > 0).sum(axis=1)
    assert widths[-1] > widths[3]                         # wider at high frequency


def test_mel_spectrogram_puts_tone_in_right_band():
    f = extract_features(tone(1000, 0.5, SR, 0.5))
    band = int(np.argmax(f["mel_spectrogram"].mean(axis=1)))
    assert f["mel_frequencies"][band] == pytest.approx(1000, rel=0.1)


def test_mfcc_shapes_and_c0_tracks_loudness(feats, speech):
    T = feats["times"].size
    for k in ("mfcc", "mfcc_delta", "mfcc_delta2"):
        assert feats[k].shape == (13, T)
    quiet = extract_features(0.1 * speech)
    # c0 = sqrt(26) * mean log energy (orthonormal DCT): a gain of 0.1 lowers it by sqrt(26)*ln(100)
    shift = (feats["mfcc"][0] - quiet["mfcc"][0])
    loud = feats["energy_db"] > np.percentile(feats["energy_db"], 50)
    assert np.median(shift[loud]) == pytest.approx(np.sqrt(26) * np.log(100), rel=0.01)
    np.testing.assert_allclose(feats["mfcc"][1:, loud], quiet["mfcc"][1:, loud], atol=5e-3)


def test_mfcc_equals_dct_of_log_mel():
    from scipy.fft import dct
    x = speech_like(1.0, SR)
    fr = F.frame_signal(x, 400, 160)
    P = F.power_spectrogram(fr, 512)
    fb, _ = F.mel_filterbank(SR, 512, 26)
    log_mel = np.log(fb @ P + 1e-10)
    np.testing.assert_allclose(F.mfcc(P, SR, 512), dct(log_mel, 2, axis=0, norm="ortho")[:13])


def test_delta_of_linear_ramp_is_slope():
    c = np.tile(np.arange(20.0) * 0.5, (3, 1))
    d = F.delta(c, 2)
    np.testing.assert_allclose(d[:, 2:-2], 0.5)
    np.testing.assert_allclose(F.delta(np.ones((2, 10))), 0.0)


def test_matches_librosa_with_identical_settings():
    librosa = pytest.importorskip("librosa")
    x = speech_like(2.0, SR).astype(np.float64)
    f = extract_features(x)
    # librosa centres a 400-sample window in each 512-sample frame: shift by 56 to align
    S = np.abs(librosa.stft(np.concatenate([np.zeros(56), x]), n_fft=512, hop_length=160,
                            win_length=400, window=np.hamming(400), center=False)) ** 2 / 512
    T = S.shape[1]
    mb = librosa.filters.mel(sr=SR, n_fft=512, n_mels=26, htk=True, norm=None)
    lm = librosa.feature.mfcc(S=np.log(mb @ S + 1e-10), n_mfcc=13, dct_type=2, norm="ortho")
    np.testing.assert_allclose(F.mel_filterbank(SR, 512, 26)[0], mb, atol=1e-6)
    np.testing.assert_allclose(f["mfcc"][:, :T], lm, atol=1e-3)
    c = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0]
    np.testing.assert_allclose(f["spectral_centroid"][:T], c, rtol=1e-4, atol=1e-3)
    y = librosa.yin(x, fmin=60, fmax=500, sr=SR, frame_length=1280, hop_length=160)
    assert np.nanmedian(f["f0"]) == pytest.approx(np.median(y), rel=0.03)


# ------------------------------------------------------------------ pitch
@pytest.mark.parametrize("f0", [80, 120, 200, 310, 450])
def test_pitch_of_harmonic_tone(f0):
    f = extract_features(harmonic(f0))
    assert f["voiced"].mean() > 0.95
    assert np.nanmedian(f["f0"]) == pytest.approx(f0, rel=0.01)


def test_pitch_tracks_glide_of_speech_like(feats):
    t, f0 = feats["times"], feats["f0"]
    v = feats["voiced"]
    true = 110 + 30 * t / 3.0                               # synth.speech_like F0 glide
    assert v.mean() > 0.4
    err = np.abs(f0[v] - true[v]) / true[v]
    assert np.median(err) < 0.03 and np.mean(err < 0.1) > 0.9


def test_pitch_survives_bandpass_missing_fundamental(speech):
    # 1B removes everything below 300 Hz, i.e. the 110-140 Hz fundamental itself;
    # YIN measures the period, which the harmonics still carry
    f = extract_features(pp.clean_audio(speech, vad=False))
    assert np.nanmedian(f["f0"][: 100]) == pytest.approx(115, rel=0.05)


def test_unvoiced_noise_and_silence_have_no_pitch():
    f = extract_features(0.3 * np.random.default_rng(2).standard_normal(SR).astype(np.float32))
    assert f["voiced"].mean() < 0.05
    s = extract_features(np.zeros(SR, np.float32))
    assert not s["voiced"].any() and np.all(np.isnan(s["f0"]))


def test_pauses_are_unvoiced():
    x = np.concatenate([np.zeros(SR // 2, np.float32), harmonic(150, 1.0),
                        1e-4 * np.random.default_rng(3).standard_normal(SR // 2).astype(np.float32)])
    f = extract_features(x)
    t = f["times"]
    assert not f["voiced"][t < 0.45].any() and not f["voiced"][t > 1.55].any()
    assert f["voiced"][(t > 0.6) & (t < 1.4)].all()


def test_voiced_prob_range(feats):
    assert np.all((feats["voiced_prob"] >= 0) & (feats["voiced_prob"] <= 1))
    assert feats["voiced_prob"][feats["voiced"]].min() > 0.8


# ------------------------------------------------------------------ API
def test_contract_keys_and_shapes(feats):
    T = feats["times"].size
    for k in ("energy", "energy_db", "zcr", "spectral_centroid", "f0", "voiced", "voiced_prob"):
        assert feats[k].shape == (T,), k
    assert feats["mel_spectrogram"].shape == (40, T)
    assert feats["mel_spectrogram_db"].shape == (40, T)
    assert feats["voiced"].dtype == bool and feats["sr"] == SR
    for k, v in feats.items():
        if isinstance(v, np.ndarray) and k != "f0":
            assert np.all(np.isfinite(v)), k


def test_accepts_paths_and_tuples():
    p = os.path.join(SAMPLES, "speech_like_48k_stereo.wav")
    a = extract_features(p)                               # via 1A: 16 kHz mono
    assert a["sr"] == SR and a["times"].size == pytest.approx(300, abs=2)
    from audio_io import load_audio
    x, _ = load_audio(p)
    b = extract_features((x, SR))
    np.testing.assert_allclose(a["mfcc"], b["mfcc"], atol=0.02)   # tuple is re-normalised by 1A


def test_other_sample_rate_array():
    x = speech_like(1.0, 8000)
    f = extract_features(x, sr=8000, n_fft=256)
    assert f["sr"] == 8000 and f["frame_len"] == 200 and f["hop"] == 80
    assert np.nanmedian(f["f0"]) == pytest.approx(125, rel=0.05)   # 110 -> 140 Hz glide


def test_short_clip_still_gives_one_frame():
    f = extract_features(np.full(100, 0.1, np.float32))
    assert f["times"].size == 1 and f["mfcc"].shape == (13, 1)


@pytest.mark.parametrize("bad", [np.zeros((2, 100), np.float32), np.zeros(0, np.float32),
                                 np.array([0.1, np.nan], np.float32)])
def test_bad_arrays_rejected(bad):
    with pytest.raises(ValueError):
        extract_features(bad)


@pytest.mark.parametrize("kw", [dict(hop_ms=30), dict(n_fft=256), dict(n_mfcc=30),
                                dict(f0_min=600), dict(pre_emphasis=1.5), dict(fmax=9000),
                                dict(yin_threshold=0)])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        extract_features(np.zeros(SR, np.float32), **kw)


def test_config_object_and_overrides(speech):
    a = extract_features(speech, config=FeatureConfig(n_mfcc=20, n_mels=64))
    assert a["mfcc"].shape[0] == 20 and a["mel_spectrogram"].shape[0] == 64


def test_input_not_modified(speech):
    x = speech.copy()
    extract_features(x, pre_emphasis=0.97)
    np.testing.assert_array_equal(x, speech)


def test_summary(feats):
    s = summarize_features(feats)
    assert s["duration_s"] == pytest.approx(3.0, abs=0.03)
    assert s["median_f0_hz"] == pytest.approx(125, rel=0.06)
    assert 0.4 < s["voiced_fraction"] < 0.7
    s0 = summarize_features(extract_features(np.zeros(SR, np.float32)))
    assert np.isnan(s0["median_f0_hz"]) and s0["voiced_fraction"] == 0


def test_speed(speech):
    import time
    t = time.perf_counter()
    extract_features(speech)
    assert time.perf_counter() - t < 1.0                  # 3 s of audio well under real time
