"""Tests for stt.py (sub-module 2A).  No network and no model needed: the cloud API is imitated by a
local HTTP server, the engines by fakes.  Real-model tests are at the bottom (skipped if unavailable)."""
import io, json, logging, os, sys, threading, time, types
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from synth import speech_like  # noqa: E402

import stt  # noqa: E402
from audio_io import AudioLoadError  # noqa: E402
from stt import (CloudEngine, CloudError, EngineResult, LocalEngine, LocalEngineError, SpeechToText,  # noqa: E402
                 STTConfig, STTError, language_from_script, normalize_language, restrict_language_probs)

ML = "ഈ നഗരം രാജ്യത്തെ മറ്റ് നഗരങ്ങളിൽ നിന്ന് വ്യത്യസ്തമാണ്"
KN = "ಬ್ಲಾಗಿಂಗ್ ಸಹಯೋಗವನ್ನು ಪ್ರೇರೇಪಿಸುವ ಒಂದು ಸಾಧನವಾಗಿದೆ"
HI = "कुछ अणुओं में अस्थिर केंद्रक होता है"
EN = "however due to the slow communication channels"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)         # never let a developer's real .env leak into the tests
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "SVARA_STT_API_KEY", "SVARA_STT_BASE_URL", "SVARA_STT_PROVIDER",
              "SVARA_STT_MODEL", "SVARA_STT_ENGINE", "SVARA_WHISPER_PATH", "SVARA_WHISPER_MODEL", "SVARA_STT_OFFLINE"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def audio():
    x = speech_like(2.0, 16000)
    return (0.9 * x / np.max(np.abs(x))).astype(np.float32)


# ============================================================ mock cloud API
class MockAPI:
    def __init__(self):
        self.requests, self.queue = [], []
        api = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
                parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True) for p in msg.iter_parts()}
                api.requests.append(dict(path=self.path, auth=self.headers.get("Authorization"), parts=parts))
                beh = api.queue.pop(0) if api.queue else dict(status=200, body=None)
                if beh.get("delay"):
                    time.sleep(beh["delay"])
                b = beh.get("body")
                if beh["status"] == 200 and b is None:
                    lang = parts.get("language", b"").decode() or "malayalam"
                    b = {"task": "transcribe", "language": lang, "duration": 2.0, "text": " " + ML + " ",
                         "segments": [{"start": 0.0, "end": 2.0, "text": ML, "avg_logprob": -0.3, "no_speech_prob": 0.01}]}
                data = b if isinstance(b, (bytes, str)) else json.dumps(b)
                data = data.encode() if isinstance(data, str) else data
                self.send_response(beh["status"]); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):      # client gave up (time-out test): not an error
                pass

        self.srv = QuietServer(("127.0.0.1", 0), H)
        self.base_url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


@pytest.fixture
def api():
    a = MockAPI(); yield a; a.close()


def cfg(api=None, **kw):
    base = dict(api_key="sk-test-SECRET123", cloud_provider="custom", base_url=api.base_url if api else None,
                cloud_model="whisper-test", retry_backoff_s=0.0, read_timeout_s=0.6, connect_timeout_s=1.0)
    base.update(kw)
    return STTConfig(**base)


# ============================================================ language helpers
@pytest.mark.parametrize("raw,code", [("Malayalam", "ml"), ("malayalam", "ml"), ("ML", "ml"), ("ml-IN", "ml"),
                                      ("kannada", "kn"), ("Hindi", "hi"), ("english", "en"), ("en_US", "en"),
                                      ("tamil", "ta"), (" Telugu ", "te"), (None, None), ("", None)])
def test_normalize_language(raw, code):
    assert normalize_language(raw) == code


def test_script_detection():
    assert language_from_script(ML) == "ml" and language_from_script(KN) == "kn"
    assert language_from_script(HI) == "hi" and language_from_script(EN) == "en"
    assert language_from_script("ok") is None                  # too short
    assert language_from_script("") is None
    assert language_from_script("hello नमस्ते") is None        # mixed, neither >= 70 %
    assert language_from_script("Dr. मेहता का फोन 123, ठीक है") == "hi"  # digits/punctuation ignored


def test_restrict_language_probs_renormalises():
    probs = [("te", 0.5), ("kn", 0.3), ("hi", 0.1), ("en", 0.05), ("ml", 0.05)]
    best, conf, p, mass = restrict_language_probs(probs)
    assert best == "kn" and conf == pytest.approx(0.3 / 0.5) and mass == pytest.approx(0.5)
    assert sum(p.values()) == pytest.approx(1.0) and set(p) == {"ml", "kn", "hi", "en"}


def test_restrict_language_probs_no_supported_mass():
    best, conf, p, mass = restrict_language_probs([("ja", 1.0)])
    assert best == "en" and conf == 0.0 and mass == 0.0


# ============================================================ cloud engine against the mock server
def test_cloud_request_format(api, audio):
    eng = CloudEngine(cfg(api))
    r = eng.transcribe(audio, language="ml", prompt="Hospital form")
    q = api.requests[0]
    assert q["path"] == "/v1/audio/transcriptions"
    assert q["auth"] == "Bearer sk-test-SECRET123"
    assert q["parts"]["model"] == b"whisper-test"
    assert q["parts"]["response_format"] == b"verbose_json"
    assert q["parts"]["language"] == b"ml" and q["parts"]["prompt"] == b"Hospital form"
    data, sr = sf.read(io.BytesIO(q["parts"]["file"]))
    assert sr == 16000 and data.ndim == 1 and len(data) == len(audio)
    assert r.text.strip() == ML and r.engine == "cloud" and r.segments[0]["avg_logprob"] == -0.3


def test_cloud_language_omitted_when_not_given(api, audio):
    CloudEngine(cfg(api)).transcribe(audio)
    assert "language" not in api.requests[0]["parts"] and "prompt" not in api.requests[0]["parts"]


@pytest.mark.parametrize("returned,code", [("malayalam", "ml"), ("Kannada", "kn"), ("hindi", "hi"), ("english", "en"),
                                           ("ml", "ml"), ("tamil", "ta")])
def test_cloud_parses_language_name_or_code(api, audio, returned, code):
    api.queue.append(dict(status=200, body={"language": returned, "text": "x y z"}))
    assert CloudEngine(cfg(api)).transcribe(audio).language == code


def test_cloud_401_not_retried(api, audio):
    api.queue.append(dict(status=401, body={"error": "bad key sk-test-SECRET123"}))
    with pytest.raises(CloudError, match="rejected the API key") as e:
        CloudEngine(cfg(api)).transcribe(audio)
    assert len(api.requests) == 1 and e.value.status == 401 and not e.value.retryable


def test_cloud_5xx_then_ok_is_retried(api, audio):
    api.queue += [dict(status=503, body="busy")]
    r = CloudEngine(cfg(api)).transcribe(audio)
    assert len(api.requests) == 2 and r.text


def test_cloud_persistent_5xx_fails(api, audio):
    api.queue += [dict(status=500, body="x"), dict(status=500, body="x")]
    with pytest.raises(CloudError, match="HTTP 500") as e:
        CloudEngine(cfg(api)).transcribe(audio)
    assert e.value.retryable and len(api.requests) == 2


def test_cloud_429_retried(api, audio):
    api.queue.append(dict(status=429, body="slow down"))
    assert CloudEngine(cfg(api)).transcribe(audio).text


def test_cloud_400_error_message_hides_key(api, audio):
    api.queue.append(dict(status=400, body="invalid file; token sk-test-SECRET123"))
    with pytest.raises(CloudError) as e:
        CloudEngine(cfg(api)).transcribe(audio)
    assert "SECRET123" not in str(e.value) and "***" in str(e.value)


def test_cloud_timeout(api, audio):
    api.queue += [dict(status=200, body=None, delay=1.5)] * 2
    with pytest.raises(CloudError, match="timed out"):
        CloudEngine(cfg(api, read_timeout_s=0.3)).transcribe(audio)


def test_cloud_connection_refused(audio):
    dead = STTConfig(api_key="k", cloud_provider="custom", base_url="http://127.0.0.1:9/v1", retry_backoff_s=0, connect_timeout_s=0.5)
    with pytest.raises(CloudError, match="network error|timed out") as e:   # Windows: connect time-out, Linux: refused
        CloudEngine(dead).transcribe(audio)
    assert e.value.retryable                                               # either way the orchestrator may fail over


def test_cloud_connect_timeout_and_refused_both_retryable(monkeypatch, audio):
    import requests
    for exc, text in [(requests.exceptions.ConnectTimeout("t"), "timed out"),
                      (requests.exceptions.ConnectionError("refused"), "network error"),
                      (requests.exceptions.ReadTimeout("t"), "timed out")]:
        eng = CloudEngine(STTConfig(api_key="k", cloud_provider="custom", base_url="https://example.com/v1", retry_backoff_s=0))
        def boom(*a, **k): raise exc
        monkeypatch.setattr(eng, "_post", boom)
        with pytest.raises(CloudError, match=text) as e:
            eng.transcribe(audio)
        assert e.value.retryable


def test_connect_failure_falls_back_to_local(audio):
    dead = STTConfig(api_key="k", cloud_provider="custom", base_url="http://127.0.0.1:9/v1", retry_backoff_s=0, connect_timeout_s=0.3)
    s = SpeechToText(dead, local=FakeEngine("local", [EngineResult(EN, "en", "local", "m")]))
    r = s.transcribe_full(audio)
    assert r.engine == "local" and r.fallback_used and r.fallback_reason.startswith("cloud:")


@pytest.mark.parametrize("body", ["not json at all", {"no_text": 1}, {"text": 123}, "[]"])
def test_cloud_bad_response(api, audio, body):
    api.queue.append(dict(status=200, body=body))
    with pytest.raises(CloudError, match="unreadable"):
        CloudEngine(cfg(api)).transcribe(audio)


def test_cloud_too_large(api):
    with pytest.raises(CloudError, match="too large"):
        CloudEngine(cfg(api, max_upload_bytes=1000)).transcribe(np.zeros(32000, np.float32))


def test_cloud_not_configured_and_https_rule(audio):
    assert not CloudEngine(STTConfig()).available
    with pytest.raises(CloudError, match="no cloud API key"):
        CloudEngine(STTConfig()).transcribe(audio)
    with pytest.raises(ValueError, match="https"):
        CloudEngine(STTConfig(api_key="k", cloud_provider="custom", base_url="http://example.com/v1"))


def test_provider_selection_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g"); e = CloudEngine(STTConfig())
    assert (e.provider, e.model, e.available) == ("groq", "whisper-large-v3", True) and "groq.com" in e.base_url
    monkeypatch.delenv("GROQ_API_KEY"); monkeypatch.setenv("OPENAI_API_KEY", "o"); e = CloudEngine(STTConfig())
    assert (e.provider, e.model) == ("openai", "whisper-1") and "openai.com" in e.base_url
    e = CloudEngine(STTConfig(cloud_provider="groq", cloud_model="whisper-large-v3-turbo", api_key="x"))
    assert e.model == "whisper-large-v3-turbo"


def test_api_key_never_logged(api, audio, caplog):
    caplog.set_level(logging.DEBUG)
    api.queue.append(dict(status=500, body="boom")); api.queue.append(dict(status=500, body="boom"))
    s = SpeechToText(cfg(api), local=FakeEngine("local", [EngineResult(EN, "en", "local", "m")]))
    s.transcribe_full(speech_like(1.0, 16000))
    assert "SECRET123" not in caplog.text


# ============================================================ fakes for the orchestrator
class FakeEngine:
    def __init__(self, name, results=None, error=None, available=True):
        self.name, self.results, self.error, self.available = name, list(results or []), error, available
        self.calls, self.model_name = [], f"fake-{name}"
        self.model = f"fake-{name}"

    def transcribe(self, audio, language=None, prompt=None):
        self.calls.append(dict(audio=audio, language=language, prompt=prompt))
        if self.error:
            raise self.error
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return EngineResult(**{**r.__dict__})

    def detect_language(self, audio):
        return ("kn", 0.8, {"ml": .1, "kn": .8, "hi": .05, "en": .05}, 0.99)


def er(text, lang, engine="cloud", **kw):
    return EngineResult(text=text, language=lang, engine=engine, model=f"fake-{engine}", **kw)


def make(cloud=None, local=None, **kw):
    c = cloud or FakeEngine("cloud", [er(EN, "en")])
    l = local or FakeEngine("local", [er(EN, "en", "local")])
    kw.setdefault("breaker_cooldown_s", 60)
    return SpeechToText(STTConfig(**kw), cloud=c, local=l), c, l


# ============================================================ orchestrator: happy path + failover
def test_cloud_used_first(audio):
    s, c, l = make()
    r = s.transcribe_full(audio)
    assert r.engine == "cloud" and not r.fallback_used and r.fallback_reason is None
    assert len(c.calls) == 1 and len(l.calls) == 0
    assert s.transcribe(audio) == (EN, "en")


def test_cloud_error_falls_over_to_local(audio):
    s, c, l = make(cloud=FakeEngine("cloud", error=CloudError("cloud busy/unavailable (HTTP 503)", True, 503)))
    r = s.transcribe_full(audio)
    assert r.engine == "local" and r.fallback_used and "HTTP 503" in r.fallback_reason
    assert len(c.calls) == 1 and len(l.calls) == 1


def test_no_api_key_goes_straight_to_local(audio):
    s, c, l = make(cloud=FakeEngine("cloud", available=False))
    r = s.transcribe_full(audio)
    assert r.engine == "local" and r.fallback_used and "no cloud API key" in r.fallback_reason
    assert len(c.calls) == 0


def test_circuit_breaker_skips_cloud_then_recovers(audio):
    s, c, l = make(cloud=FakeEngine("cloud", error=CloudError("net down", True)), breaker_failures=2, breaker_cooldown_s=0.3)
    s.transcribe_full(audio); s.transcribe_full(audio)            # 2 failures -> breaker opens
    assert len(c.calls) == 2
    r = s.transcribe_full(audio)                                  # skipped, instantly
    assert len(c.calls) == 2 and r.engine == "local" and "skipped after repeated failures" in r.fallback_reason
    time.sleep(0.35)
    s.transcribe_full(audio)                                      # cooldown over -> cloud tried again
    assert len(c.calls) == 3


def test_breaker_resets_after_success(audio):
    c = FakeEngine("cloud", [er(EN, "en")])
    s, c, l = make(cloud=c, breaker_failures=2)
    s._cloud_failed(); s.transcribe_full(audio)                   # success clears the counter
    s._cloud_failed()
    assert s._cloud_open_until == 0.0


def test_missing_key_does_not_trip_breaker(audio):
    s, c, l = make(cloud=FakeEngine("cloud", available=False), breaker_failures=1)
    for _ in range(3):
        s.transcribe_full(audio)
    assert s._cloud_failures == 0


def test_both_engines_fail(audio):
    s, c, l = make(cloud=FakeEngine("cloud", error=CloudError("a")), local=FakeEngine("local", error=LocalEngineError("b")))
    with pytest.raises(STTError) as e:
        s.transcribe_full(audio)
    assert "cloud: a" in str(e.value) and "local: b" in str(e.value)


def test_forced_engines(audio):
    s, c, l = make()
    assert s.transcribe_full(audio, engine="local").engine == "local" and not c.calls
    r = s.transcribe_full(audio, engine="cloud"); assert r.engine == "cloud" and not r.fallback_used
    s2, c2, l2 = make(cloud=FakeEngine("cloud", error=CloudError("down")))
    with pytest.raises(STTError):                                   # no silent fallback when forced
        s2.transcribe_full(audio, engine="cloud")
    assert not l2.calls
    with pytest.raises(ValueError):
        s.transcribe_full(audio, engine="magic")


# ============================================================ language handling
def test_language_hint_is_passed_and_normalised(audio):
    s, c, l = make(cloud=FakeEngine("cloud", [er(ML, "ml")]))
    r = s.transcribe_full(audio, language="Malayalam", prompt="form")
    assert c.calls[0]["language"] == "ml" and c.calls[0]["prompt"] == "form" and r.language == "ml"
    with pytest.raises(ValueError, match="not supported"):
        s.transcribe_full(audio, language="tamil")
    with pytest.raises(ValueError):
        s.transcribe_full(audio, language="xx")


def test_unsupported_cloud_language_retried_with_confusion_map(audio):
    c = FakeEngine("cloud", [er("தமிழ் text", "ta"), er(ML, "ml")])
    s, c, l = make(cloud=c)
    r = s.transcribe_full(audio)
    assert [x["language"] for x in c.calls] == [None, "ml"]       # Tamil -> Malayalam
    assert r.language == "ml" and r.text == ML and any("not supported" in w for w in r.warnings)


def test_unsupported_language_uses_script_when_available(audio):
    c = FakeEngine("cloud", [er(KN, "te"), er(KN, "kn")])
    s, c, l = make(cloud=c)
    s.transcribe_full(audio)
    assert c.calls[1]["language"] == "kn"                          # script says Kannada


def test_unknown_unrelated_language_defaults_to_english(audio):
    c = FakeEngine("cloud", [er("bonjour tout le monde", "fr"), er("hello everyone", "en")])
    s, c, l = make(cloud=c); r = s.transcribe_full(audio)
    assert c.calls[1]["language"] == "en" and r.language == "en"


def test_script_corrects_wrong_language_label(audio):
    s, c, l = make(cloud=FakeEngine("cloud", [er(ML, "hi")]))
    r = s.transcribe_full(audio)
    assert r.language == "ml" and any("corrected" in w for w in r.warnings)


def test_latin_text_with_indic_label_warns_but_keeps_label(audio):
    s, c, l = make(cloud=FakeEngine("cloud", [er("This city is completely different", "ml")]))
    r = s.transcribe_full(audio)
    assert r.language == "ml" and any("Latin letters" in w for w in r.warnings)


def test_hint_with_mismatching_script_warns(audio):
    s, c, l = make(cloud=FakeEngine("cloud", [er(HI, "kn")]))
    r = s.transcribe_full(audio, language="kn")
    assert r.language == "kn" and any("looks like 'hi'" in w for w in r.warnings)


def test_low_confidence_and_low_mass_warnings(audio):
    s, c, l = make(local=FakeEngine("local", [er(HI, "hi", "local", language_confidence=0.4, supported_mass=0.1)]),
                   cloud=FakeEngine("cloud", available=False))
    r = s.transcribe_full(audio)
    assert any("Low language confidence" in w for w in r.warnings) and any("other than" in w for w in r.warnings)
    assert r.language_confidence == 0.4


# ============================================================ text safeguards
def test_text_is_nfc_and_whitespace_normalised(audio):
    decomposed = "é   cole\n\tbon"                         # e + combining acute
    s, *_ = make(cloud=FakeEngine("cloud", [er(decomposed, "en")]))
    assert s.transcribe(audio)[0] == "é cole bon"


def test_repetition_loop_discarded(audio):
    s, *_ = make(cloud=FakeEngine("cloud", [er("thank you " * 20, "en")]))
    r = s.transcribe_full(audio)
    assert r.text == "" and any("repeating loop" in w for w in r.warnings)
    s, *_ = make(cloud=FakeEngine("cloud", [er("അഅഅഅഅഅഅഅഅഅഅഅഅഅഅഅഅ", "ml")]))
    assert s.transcribe(audio)[0] == ""


def test_normal_repetition_not_discarded(audio):
    s, *_ = make(cloud=FakeEngine("cloud", [er("one two three one two three four", "en")]))
    assert s.transcribe(audio)[0] == "one two three one two three four"


def test_empty_text_flags_and_keeps_language_unknown_when_unsupported(audio):
    s, *_ = make(cloud=FakeEngine("cloud", [er("", "ml")]))
    r = s.transcribe_full(audio)
    assert r.text == "" and r.language == "ml" and any("No speech recognised" in w for w in r.warnings)


# ============================================================ input handling
def test_silence_and_short_audio_never_reach_engines():
    s, c, l = make()
    r = s.transcribe_full(np.zeros(32000, np.float32))
    assert r.skipped == "silence" and r.text == "" and r.language == "unknown" and not c.calls and not l.calls
    r = s.transcribe_full(np.full(800, 0.5, np.float32))           # 50 ms
    assert r.skipped == "too short" and not c.calls
    r = s.transcribe_full(np.zeros(32000, np.float32), language="hi")
    assert r.language == "hi"                                        # hint survives


def test_clean_1b_array_passed_through_untouched(audio):
    s, c, l = make()
    s.transcribe_full(audio)
    assert c.calls[0]["audio"] is audio or np.array_equal(c.calls[0]["audio"], audio)
    assert c.calls[0]["audio"].dtype == np.float32


def test_input_types_file_bytes_tuple_and_resampling(tmp_path):
    x44 = speech_like(2.0, 44100)
    p = tmp_path / "a.wav"; sf.write(p, x44, 44100)
    raw = p.read_bytes()
    for src, sr in [(str(p), None), (p, None), (raw, None), (io.BytesIO(raw), None), ((x44, 44100), None), (x44, 44100)]:
        s, c, l = make()
        assert s.transcribe(src, sr)[1] == "en"
        a = c.calls[0]["audio"]
        assert a.ndim == 1 and a.dtype == np.float32 and abs(len(a) - 32000) <= 1     # 16 kHz after 1A


def test_bad_audio_errors_are_1a_errors():
    s, *_ = make()
    for bad in (b"not audio", "/no/such/file.wav", np.array([np.nan] * 16000, np.float32)):
        with pytest.raises(AudioLoadError):
            s.transcribe_full(bad)


def test_preprocess_flag_runs_1b(audio):
    noisy = np.concatenate([np.zeros(16000), audio, np.zeros(16000)]).astype(np.float32)
    noisy += 0.003 * np.random.default_rng(0).standard_normal(len(noisy)).astype(np.float32)
    s, c, l = make()
    s.transcribe_full(noisy, preprocess=True)
    assert len(c.calls[0]["audio"]) < len(noisy) - 8000                  # silence was trimmed by 1B


def test_module_level_api(monkeypatch, audio):
    s, c, l = make()
    monkeypatch.setattr(stt, "_default", s)
    assert stt.transcribe(audio) == (EN, "en")
    assert stt.transcribe_full(audio).engine == "cloud"
    assert stt.detect_language(audio)[0] == "kn"                          # fake local LID
    stt.reset_default(); assert stt._default is None


def test_detect_language_falls_back_to_cloud(audio):
    s, c, l = make(local=FakeEngine("local", available=False), cloud=FakeEngine("cloud", [er(ML, "ml")]))
    assert s.detect_language(audio)[0] == "ml"


# ============================================================ local engine with a fake faster_whisper
class _Seg:
    def __init__(self, text, no_speech=0.01, lp=-0.3):
        self.start, self.end, self.text, self.avg_logprob, self.no_speech_prob = 0.0, 1.0, text, lp, no_speech


def fake_fw(lang_info, texts, load_error=None):
    """A stand-in for the faster_whisper module that records its calls."""
    calls = types.SimpleNamespace(loads=[], runs=[], detects=[])
    class FakeModel:
        def __init__(self, name, **kw):
            if load_error: raise load_error
            calls.loads.append((name, kw))
        def transcribe(self, audio, language=None, **kw):
            calls.runs.append(dict(language=language, **kw))
            info = types.SimpleNamespace(language=lang_info[0], all_language_probs=lang_info[1] if language is None else None,
                                         duration=len(audio) / 16000)
            return iter([_Seg(texts[language or lang_info[0]])]), info
        def detect_language(self, audio=None, **kw):
            calls.detects.append(1)
            return lang_info[0], 0.9, lang_info[1]
    mod = types.ModuleType("faster_whisper"); mod.WhisperModel = FakeModel
    return mod, calls


def test_local_engine_restricts_language_and_reruns(monkeypatch, audio):
    # Whisper's top guess is Telugu (outside the four); Kannada is the best supported one
    probs = [("te", 0.55), ("kn", 0.30), ("hi", 0.10), ("ml", 0.03), ("en", 0.02)]
    mod, calls = fake_fw(("te", probs), {"te": "తెలుగు", "kn": KN})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    r = LocalEngine(STTConfig(local_device="cpu")).transcribe(audio)
    assert r.language == "kn" and r.text == KN
    assert [x["language"] for x in calls.runs] == [None, "kn"]            # detected, then forced
    assert r.language_confidence == pytest.approx(0.30 / 0.45) and r.supported_mass == pytest.approx(0.45)
    assert any("restricted" in n for n in r.notes)


def test_local_engine_single_pass_when_best_is_supported(monkeypatch, audio):
    probs = [("hi", 0.9), ("kn", 0.05), ("en", 0.03), ("ml", 0.02)]
    mod, calls = fake_fw(("hi", probs), {"hi": HI})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    r = LocalEngine(STTConfig(local_device="cpu")).transcribe(audio)
    assert r.language == "hi" and len(calls.runs) == 1


def test_local_engine_hint_skips_detection_and_settings(monkeypatch, audio):
    mod, calls = fake_fw(("hi", [("hi", 1.0)]), {"ml": ML})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    eng = LocalEngine(STTConfig(local_device="cpu", beam_size=3))
    r = eng.transcribe(audio, language="ml", prompt="form")
    run = calls.runs[0]
    assert r.language == "ml" and run["language"] == "ml" and run["beam_size"] == 3
    assert run["task"] == "transcribe" and run["initial_prompt"] == "form" and run["vad_filter"] is False
    assert run["condition_on_previous_text"] is False and run["temperature"] == 0.0   # no slow retry loop
    name, kw = calls.loads[0]
    assert name == "small" and kw["compute_type"] == "int8" and kw["device"] == "cpu"
    eng.transcribe(audio, language="ml"); assert len(calls.loads) == 1     # model loaded once


def test_local_engine_detect_language(monkeypatch, audio):
    mod, calls = fake_fw(("ta", [("ta", .6), ("ml", .3), ("en", .1)]), {})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    best, conf, probs, mass = LocalEngine(STTConfig(local_device="cpu")).detect_language(audio)
    assert best == "ml" and conf == pytest.approx(0.75) and mass == pytest.approx(0.4)


def test_local_engine_missing_library(monkeypatch, audio):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    eng = LocalEngine(STTConfig())
    assert not eng.available
    with pytest.raises(LocalEngineError, match="pip install faster-whisper"):
        eng.transcribe(audio)


def test_local_engine_load_failure_mentions_download(monkeypatch, audio):
    mod, _ = fake_fw(("en", []), {}, load_error=OSError("offline, no cached model"))
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    with pytest.raises(LocalEngineError, match="--download"):
        LocalEngine(STTConfig(local_device="cpu")).transcribe(audio)


def test_local_files_only_and_model_path_from_env(monkeypatch):
    monkeypatch.setenv("SVARA_STT_OFFLINE", "1"); monkeypatch.setenv("SVARA_WHISPER_PATH", "/models/small")
    c = STTConfig.from_env()
    assert c.local_files_only and c.local_model == "/models/small"
    with pytest.raises(TypeError):
        STTConfig.from_env(nonsense=1)


# ============================================================ full stack: orchestrator + real classes + mock server
def test_full_stack_failover_over_http(api, monkeypatch, audio):
    mod, calls = fake_fw(("en", [("en", 0.95), ("hi", 0.05)]), {"en": EN})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    c = cfg(api, local_device="cpu", cloud_retries=0)
    s = SpeechToText(c)
    r = s.transcribe_full(audio)                                       # cloud answers (Malayalam text from mock)
    assert r.engine == "cloud" and r.language == "ml" and r.text == ML
    api.queue.append(dict(status=500, body="down"))
    r = s.transcribe_full(audio)                                       # cloud 500 -> local fake
    assert r.engine == "local" and r.fallback_used and r.text == EN and "HTTP 500" in r.fallback_reason


# ============================================================ REAL model tests (skipped unless available)
TESTDATA = os.environ.get("SVARA_STT_TESTDATA")


@pytest.fixture(scope="module")
def real_stt():
    if not TESTDATA or not os.path.exists(os.path.join(TESTDATA, "manifest.csv")):
        pytest.skip("set SVARA_STT_TESTDATA to a folder made by scripts/fetch_fleurs_samples.py")
    try:
        s = SpeechToText(STTConfig(local_files_only=True))
        s.warm_up()
    except Exception as exc:                                           # noqa: BLE001
        pytest.skip(f"Whisper small not cached ({exc}); run: python stt.py --download")
    return s


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_real_whisper_small_english_and_hindi(real_stt, lang):
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(TESTDATA, "manifest.csv"), encoding="utf-8")) if r["lang"] == lang][:2]
    for r in rows:
        res = real_stt.transcribe_full(os.path.join(TESTDATA, r["file"]), engine="local")
        assert res.language == lang and len(res.text) > 20 and res.engine == "local"


# ============================================================ reliability flag and weak-model warning
def test_reliable_true_for_clean_result(audio):
    s, *_ = make(cloud=FakeEngine("cloud", [er(EN, "en")]))
    r = s.transcribe_full(audio)
    assert r.reliable and r.warnings == []


@pytest.mark.parametrize("text,label,hint", [
    (ML, "hi", None),                                  # language corrected from script
    ("This city is different", "ml", None),            # Latin text with an Indic label
    ("This city is different", "ml", "ml"),            # forced Malayalam but got English (translation)
    (HI, "kn", "kn"),                                  # script mismatch with hint
    ("", "ml", None),                                  # nothing recognised
    ("thank you " * 20, "en", None),                   # repetition loop
])
def test_reliable_false_cases(audio, text, label, hint):
    s, *_ = make(cloud=FakeEngine("cloud", [er(text, label)]))
    r = s.transcribe_full(audio, language=hint)
    assert not r.reliable and r.warnings


def test_silence_is_not_reliable():
    s, *_ = make()
    assert not s.transcribe_full(np.zeros(32000, np.float32)).reliable


@pytest.mark.parametrize("model,lang,expect_warning", [
    ("small", "ml", True), ("small", "kn", True), ("base", "ml", True), ("/models/small", "kn", True),
    ("small", "hi", False), ("small", "en", False), ("large-v3", "ml", False), ("medium", "kn", False)])
def test_weak_local_model_warning(audio, model, lang, expect_warning):
    text = {"ml": ML, "kn": KN, "hi": HI, "en": EN}[lang]
    res = er(text, lang, "local"); res.model = f"faster-whisper:{model}"       # the model that actually ran
    s, *_ = make(cloud=FakeEngine("cloud", available=False), local=FakeEngine("local", [res]), local_model=model)
    r = s.transcribe_full(audio)
    assert any("unreliable for" in w for w in r.warnings) == expect_warning
    assert r.reliable == (not expect_warning)


def test_weak_model_warning_not_shown_for_cloud(audio):
    s, *_ = make(cloud=FakeEngine("cloud", [er(ML, "ml")]), local_model="small")
    assert s.transcribe_full(audio).reliable


def test_temperature_fallback_can_be_enabled(monkeypatch, audio):
    mod, calls = fake_fw(("hi", [("hi", 1.0)]), {"hi": HI})
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    LocalEngine(STTConfig(local_device="cpu", local_temperature_fallback=True)).transcribe(audio, language="hi")
    assert "temperature" not in calls.runs[0]                      # library default fallback list


# ============================================================ specialist (per-language) models
def test_specialist_used_for_hinted_language_before_cloud(audio):
    s, c, l = make(cloud=FakeEngine("cloud", [er(EN, "en")]), local=FakeEngine("local", [er(ML, "ml", "local")]),
                   specialists={"ml": "my/ml-model"})
    r = s.transcribe_full(audio, language="ml")
    assert r.engine == "local" and r.text == ML and not r.fallback_used and not c.calls


def test_specialist_not_used_for_other_languages(audio):
    s, c, l = make(specialists={"ml": "my/ml-model"})
    r = s.transcribe_full(audio, language="en")
    assert r.engine == "cloud" and not l.calls


def test_specialist_replaces_cloud_result_after_detection(audio):
    cloud = FakeEngine("cloud", [er("Latin garbage", "ml")])
    local = FakeEngine("local", [er(ML, "ml", "local")])
    s, c, l = make(cloud=cloud, local=local, specialists={"ml": "my/ml-model"})
    r = s.transcribe_full(audio)
    assert r.engine == "local" and r.text == ML and r.language == "ml" and not r.fallback_used
    assert l.calls[0]["language"] == "ml" and len(c.calls) == 1
    assert r.reliable and any("specialist" in w for w in r.warnings)


def test_specialist_failure_keeps_cloud_result_and_flags_it(audio):
    cloud = FakeEngine("cloud", [er(ML, "ml")])
    local = FakeEngine("local", error=LocalEngineError("model missing"))
    s, c, l = make(cloud=cloud, local=local, specialists={"ml": "my/ml-model"})
    r = s.transcribe_full(audio)
    assert r.engine == "cloud" and r.text == ML and not r.reliable
    assert any("Specialist model" in w and "model missing" in w for w in r.warnings)


def test_specialist_hint_falls_back_to_cloud_if_local_fails(audio):
    s, c, l = make(cloud=FakeEngine("cloud", [er(ML, "ml")]), local=FakeEngine("local", error=LocalEngineError("x")),
                   specialists={"ml": "my/ml-model"})
    r = s.transcribe_full(audio, language="ml")
    assert r.engine == "cloud" and r.fallback_used and "x" in r.fallback_reason


def test_specialist_not_used_when_engine_forced_to_cloud(audio):
    s, c, l = make(specialists={"ml": "my/ml-model"}, cloud=FakeEngine("cloud", [er(ML, "ml")]))
    r = s.transcribe_full(audio, engine="cloud")
    assert r.engine == "cloud" and not l.calls


def test_local_engine_loads_specialist_model_and_routes(monkeypatch, audio):
    seen = {"loads": [], "runs": []}
    class M:
        def __init__(self, name, **kw): seen["loads"].append(name); self.name = name
        def transcribe(self, a, language=None, **kw):
            seen["runs"].append((self.name, language))
            info = types.SimpleNamespace(language="ml", all_language_probs=[("ml", .7), ("kn", .2), ("hi", .1)] if language is None else None, duration=2.0)
            return iter([_Seg(ML if self.name == "spec/ml" else "garbage")]), info
    mod = types.ModuleType("faster_whisper"); mod.WhisperModel = M
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    eng = LocalEngine(STTConfig(local_device="cpu", specialists={"ml": "spec/ml"}))
    r = eng.transcribe(audio)                                         # auto: default model detects, specialist decodes
    assert seen["runs"] == [("small", None), ("spec/ml", "ml")] and r.text == ML
    assert r.model == "faster-whisper:spec/ml" and any("specialist" in n for n in r.notes)
    seen["runs"].clear()
    r = eng.transcribe(audio, language="ml")                          # hint: specialist only
    assert seen["runs"] == [("spec/ml", "ml")]
    r = eng.transcribe(audio, language="hi")                          # other language: default model
    assert seen["runs"][-1] == ("small", "hi")
    assert seen["loads"].count("small") == 1 and seen["loads"].count("spec/ml") == 1


def test_specialist_model_is_not_flagged_as_weak(audio):
    s, c, l = make(cloud=FakeEngine("cloud", available=False),
                   local=FakeEngine("local", [er(ML, "ml", "local", )]), local_model="small")
    l.results[0].model = "faster-whisper:rontroy/whisper-large-v3-malayalam-ct2"
    assert s.transcribe_full(audio).reliable
    l.results[0].model = "faster-whisper:small"
    assert not s.transcribe_full(audio).reliable


def test_specialists_from_env(monkeypatch):
    monkeypatch.setenv("SVARA_WHISPER_MODEL_ML", "rontroy/whisper-large-v3-malayalam-ct2")
    assert STTConfig.from_env().specialists == {"ml": "rontroy/whisper-large-v3-malayalam-ct2"}
