import io, os, shutil, sys, types
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from synth import speech_like, tone  # noqa: E402

import audio_io  # noqa: E402
from audio_io import (AudioLoadError, load_audio, load_audio_with_report,  # noqa: E402
                      normalize_amplitude, record_from_mic, resample_audio, save_wav)

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "sample_audio")


def dom_freq(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spec) * sr / len(x)


def wav_bytes(x, sr, subtype="PCM_16", fmt="WAV"):
    b = io.BytesIO()
    sf.write(b, x, sr, subtype=subtype, format=fmt)
    return b.getvalue()


# ---------------------------------------------------------------- output contract
@pytest.mark.parametrize("sr_in", [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000])
def test_contract_for_many_rates(sr_in):
    x = tone(440, 2.0, sr_in)
    y, sr = load_audio(x, sr_in)
    assert sr == 16000
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - 2 * 16000) <= 1
    assert np.max(np.abs(y)) == pytest.approx(0.9, abs=1e-4)
    assert abs(dom_freq(y, sr) - 440) < 2          # frequency preserved


def test_stereo_to_mono_average():
    l = tone(300, 1, 16000, 0.4); r = tone(300, 1, 16000, 0.2)
    y, _ = load_audio(np.stack([l, r], 1), 16000, normalize=None)
    assert y.shape == (16000,)
    np.testing.assert_allclose(y, 0.3 * np.sin(2*np.pi*300*np.arange(16000)/16000), atol=1e-6)


def test_channels_first_layout_accepted():
    x = np.stack([tone(300, 1, 16000), tone(300, 1, 16000)])  # (2, n)
    y, _ = load_audio(x, 16000)
    assert y.shape == (16000,)


def test_no_op_resample_when_already_16k():
    x = speech_like(2.0, 16000)
    y, _, rep = load_audio_with_report(x, 16000, normalize=None, remove_dc=False)
    assert not rep.resampled
    np.testing.assert_array_equal(y, x)


def test_antialiasing_10k_tone_is_removed_not_folded():
    # 10 kHz is above the 8 kHz Nyquist limit of 16 kHz audio. Naive decimation folds it
    # to 6 kHz; a proper resampler removes it.
    x = tone(10000, 1.0, 44100, 0.5)
    y, _ = load_audio(x, 44100, normalize=None)
    assert np.max(np.abs(y[2000:-2000])) < 0.01   # > 34 dB attenuation


def test_speech_band_passes_resampler_flat():
    for f in (300, 1000, 3000, 6000):
        y, _ = load_audio(tone(f, 1.0, 48000, 0.5), 48000, normalize=None)
        assert np.max(np.abs(y[2000:-2000])) == pytest.approx(0.5, abs=0.01)


def test_scipy_fallback_matches(monkeypatch):
    x = tone(1000, 1.0, 44100)
    a = resample_audio(x, 44100, 16000)
    monkeypatch.setattr(audio_io, "soxr", None)
    b = resample_audio(x, 44100, 16000)
    assert abs(len(a) - len(b)) <= 1
    n = min(len(a), len(b))
    assert np.max(np.abs(a[1000:n-1000] - b[1000:n-1000])) < 0.02


# ---------------------------------------------------------------- DC / normalisation
def test_dc_offset_removed():
    x = 0.3 * tone(500, 1, 16000, 1.0) + 0.4
    y, _, rep = load_audio_with_report(x.astype(np.float32), 16000, normalize=None)
    assert abs(np.mean(y)) < 1e-6
    assert rep.dc_offset_removed == pytest.approx(0.4, abs=1e-3)


def test_dc_removal_can_be_disabled():
    x = (0.1 * tone(500, 1, 16000, 1.0) + 0.4).astype(np.float32)
    y, _ = load_audio(x, 16000, normalize=None, remove_dc=False)
    assert np.mean(y) == pytest.approx(0.4, abs=1e-3)


def test_peak_normalisation_is_scale_invariant():
    base = speech_like(2.0, 16000)
    outs = [load_audio(g * base, 16000)[0] for g in (0.01, 0.1, 1.0)]
    for o in outs[1:]:
        np.testing.assert_allclose(o, outs[0], atol=2e-4)


def test_rms_mode_hits_target_without_clipping():
    y, _ = load_audio(speech_like(2.0, 16000), 16000, normalize="rms")
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    assert np.max(np.abs(y)) <= 0.99 + 1e-6
    assert rms == pytest.approx(0.1, rel=0.01)


def test_rms_mode_limits_gain_to_avoid_clipping():
    x = np.zeros(16000, np.float32); x[::1000] = 1.0; x += 0.01 * tone(300, 1, 16000, 1.0)
    y, _ = load_audio(x, 16000, normalize='rms', remove_dc=False)
    assert np.max(np.abs(y)) <= 0.99 + 1e-6


def test_normalize_none_keeps_level():
    x = 0.25 * tone(500, 1, 16000, 1.0)
    y, _ = load_audio(x, 16000, normalize=None)
    assert np.max(np.abs(y)) == pytest.approx(0.25, abs=1e-4)


def test_silence_not_amplified():
    y, _, rep = load_audio_with_report(np.zeros(16000, np.float32), 16000)
    assert rep.is_silent and np.all(y == 0) and rep.gain_applied == 1.0
    assert any("silent" in w for w in rep.warnings)


def test_near_silence_dither_flagged_as_silent():
    x = (1e-5 * np.random.default_rng(1).standard_normal(16000)).astype(np.float32)
    y, _, rep = load_audio_with_report(x, 16000)
    assert rep.is_silent and np.max(np.abs(y)) < 1e-4


def test_very_quiet_warns_about_gain():
    x = 0.0005 * tone(500, 1, 16000, 1.0)
    _, _, rep = load_audio_with_report(x, 16000)
    assert rep.gain_db > 40 and any("quiet" in w for w in rep.warnings)


def test_clipping_warning():
    x = np.clip(3 * tone(200, 1, 16000, 1.0), -1, 1)
    _, _, rep = load_audio_with_report(x, 16000)
    assert rep.clipped_fraction_in > 0.1 and any("clipped" in w for w in rep.warnings)


def test_output_never_exceeds_unit_range():
    for sr in (8000, 44100, 48000):
        x = np.sign(tone(97, 1.0, sr, 1.0))  # square wave: worst case for filter overshoot
        y, _ = load_audio(x.astype(np.float32), sr, normalize=None)
        assert np.max(np.abs(y)) <= 1.0
    with pytest.raises(ValueError):
        normalize_amplitude(np.ones(10, np.float32), "bogus")


# ---------------------------------------------------------------- input types
def test_wav_path_str_and_pathlib(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, speech_like(1.0, 22050), 22050)
    a, _ = load_audio(str(p)); b, _ = load_audio(p)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 16000


def test_int16_array_scaled():
    x = (tone(500, 1, 16000, 0.5) * 32767).astype(np.int16)
    y, _ = load_audio(x, 16000, normalize=None)
    assert np.max(np.abs(y)) == pytest.approx(0.5, abs=1e-3)


def test_uint8_array_scaled():
    x = ((tone(500, 1, 16000, 0.5) * 127) + 128).astype(np.uint8)
    y, _ = load_audio(x, 16000, normalize=None)
    assert np.max(np.abs(y)) == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT"])
def test_bytes_and_filelike_all_wav_depths(subtype):
    x = speech_like(1.0, 44100)
    raw = wav_bytes(x, 44100, subtype)
    ref, _ = load_audio(raw)
    assert len(ref) == 16000
    for src in (io.BytesIO(raw), bytearray(raw), memoryview(raw)):
        y, _ = load_audio(src)
        np.testing.assert_allclose(y, ref, atol=1e-6)


def test_streamlit_like_uploaded_file():
    class Uploaded(io.BytesIO):  # Streamlit's UploadedFile subclasses BytesIO
        pass
    f = Uploaded(wav_bytes(speech_like(1.0, 48000), 48000))
    f.read()                      # pointer at EOF, as after a previous read
    y, _, rep = load_audio_with_report(f)
    assert len(y) == 16000 and rep.source_type == "file-like"


def test_flac_and_ogg_via_soundfile(tmp_path):
    for ext in ("flac", "ogg"):
        p = tmp_path / f"a.{ext}"
        sf.write(p, speech_like(1.0, 16000), 16000)
        y, _, rep = load_audio_with_report(p)
        assert len(y) == 16000 and rep.decoder == "soundfile"


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg not installed (needed for WebM/M4A only)")


@pytest.mark.parametrize("name", [
    "speech_like.mp3",
    pytest.param("speech_like.webm", marks=needs_ffmpeg),
    pytest.param("speech_like.m4a", marks=needs_ffmpeg),
])
def test_compressed_formats_decode(name):
    path = os.path.join(SAMPLES, name)
    y, sr, rep = load_audio_with_report(path)
    assert sr == 16000 and abs(len(y) - 48000) < 3000   # codec padding tolerance
    assert rep.decoder in ("soundfile", "ffmpeg")
    with open(path, "rb") as fh:                         # same via bytes (browser upload)
        y2, _ = load_audio(fh.read())
    assert abs(len(y2) - len(y)) <= 1


def test_tuple_both_orders():
    x = tone(440, 1, 8000)
    a, _ = load_audio((x, 8000)); b, _ = load_audio((8000, x))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("bad", [
    b"", b"this is not audio at all", 12345, None, {"a": 1},
])
def test_bad_inputs_raise_audioloaderror(bad):
    with pytest.raises(AudioLoadError):
        load_audio(bad)


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(tmp_path / "nope.wav")
    with pytest.raises(AudioLoadError, match="Not a file"):
        load_audio(tmp_path)


def test_text_file_with_wav_extension(tmp_path):
    p = tmp_path / "fake.wav"; p.write_text("hello world")
    with pytest.raises(AudioLoadError):
        load_audio(p)


def test_array_without_sample_rate():
    with pytest.raises(AudioLoadError, match="sample_rate"):
        load_audio(np.zeros(100, np.float32))


@pytest.mark.parametrize("sr", [0, -5, 10, 1e9, float("nan"), 44100.5, "abc"])
def test_bad_sample_rate(sr):
    with pytest.raises(AudioLoadError):
        load_audio(np.ones(1000, np.float32), sr)


def test_nan_inf_rejected():
    for bad in (np.nan, np.inf):
        x = tone(300, 1, 16000); x[10] = bad
        with pytest.raises(AudioLoadError, match="NaN"):
            load_audio(x, 16000)


def test_empty_and_too_short():
    with pytest.raises(AudioLoadError):
        load_audio(np.zeros(0, np.float32), 16000)
    with pytest.raises(AudioLoadError, match="too short"):
        load_audio(np.ones(10, np.float32), 16000)


def test_too_long_rejected_for_bytes_and_array():
    with pytest.raises(AudioLoadError, match="limit"):
        load_audio(wav_bytes(tone(300, 3, 8000), 8000), max_duration_s=2)
    with pytest.raises(AudioLoadError, match="limit"):
        load_audio(tone(300, 3, 8000), 8000, max_duration_s=2)
    y, _ = load_audio(tone(300, 3, 8000), 8000, max_duration_s=None)
    assert len(y) == 48000


def test_bad_arrays():
    with pytest.raises(AudioLoadError):
        load_audio(np.zeros((2, 2, 2), np.float32), 16000)
    with pytest.raises(AudioLoadError):
        load_audio(np.array([1 + 1j] * 100), 16000)
    with pytest.raises(AudioLoadError):
        load_audio(np.zeros((16000, 100), np.float32), 16000)


def test_text_mode_filelike_rejected():
    with pytest.raises(AudioLoadError):
        load_audio(io.StringIO("text"))


def test_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(audio_io.shutil, "which", lambda _: None)
    with pytest.raises(AudioLoadError, match="ffmpeg"):
        load_audio(b"not wav data, force fallback")


# ---------------------------------------------------------------- mic + save
def _fake_sd(rates_ok=(16000,), native=44100):
    m = types.ModuleType("sounddevice")
    calls = []
    def rec(frames, samplerate, channels, dtype, device=None):
        if samplerate not in rates_ok:
            raise RuntimeError("Invalid sample rate")
        calls.append(samplerate)
        t = np.arange(frames) / samplerate
        return (0.2 * np.sin(2*np.pi*440*t))[:, None].astype(np.float32)
    m.rec, m.wait, m.calls = rec, (lambda: None), calls
    m.query_devices = lambda device=None, kind=None: {"default_samplerate": native}
    return m


def test_microphone_16k(monkeypatch):
    fake = _fake_sd(); monkeypatch.setitem(sys.modules, "sounddevice", fake)
    y, sr = record_from_mic(2.0)
    assert sr == 16000 and len(y) == 32000 and fake.calls == [16000]
    assert abs(dom_freq(y, sr) - 440) < 2


def test_microphone_falls_back_to_native_rate(monkeypatch):
    fake = _fake_sd(rates_ok=(44100,)); monkeypatch.setitem(sys.modules, "sounddevice", fake)
    y, sr = record_from_mic(2.0)
    assert sr == 16000 and abs(len(y) - 32000) <= 1 and fake.calls == [44100]


def test_microphone_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import raises ImportError
    with pytest.raises(AudioLoadError, match="sounddevice"):
        record_from_mic(1.0)
    with pytest.raises(AudioLoadError):
        record_from_mic(-1)


def test_save_wav_roundtrip(tmp_path):
    y, sr = load_audio(speech_like(1.0, 44100), 44100)
    p = save_wav(y, tmp_path / "o.wav", sr)
    info = sf.info(p)
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    z, _ = load_audio(p, normalize=None)
    np.testing.assert_allclose(z, y, atol=1e-4)
    with pytest.raises(AudioLoadError):
        save_wav(np.zeros((10, 2), np.float32), tmp_path / "x.wav")


# ---------------------------------------------------------------- shipped samples
SAMPLE_WAVS = ["alias_test_10k_tone_44k1.wav", "silence_16k.wav", "speech_like_44k1_mono.wav",
               "speech_like_48k_stereo.wav", "speech_like_8k_phone.wav", "speech_like_dc_offset_16k.wav",
               "speech_like_loud_clipped_16k.wav", "speech_like_quiet_16k.wav"]


@pytest.mark.parametrize("name", SAMPLE_WAVS)
def test_all_sample_wavs_load(name):
    y, sr, rep = load_audio_with_report(os.path.join(SAMPLES, name))
    assert sr == 16000 and y.ndim == 1 and np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 0.9 + 1e-4
    assert rep.is_silent == (name == "silence_16k.wav")
