"""Tests for 2C tts.py.  A fake engine is used, so no internet is needed.
Set SVARA_ONLINE_TESTS=1 to also run one real gTTS request per language."""
import os

import pytest

import tts as T

FAKE_MP3 = b"ID3" + b"\x00" * 64


class FakeBackend:
    name = "fake"

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def cache_tag(self, language, slow):
        return f"fake|{language}|{slow}"

    def synthesize(self, text, language, slow=False):
        self.calls.append((text, language, slow))
        if self.fail:
            raise T.TTSError("offline")
        return FAKE_MP3 + f"{language}:{text}".encode("utf-8")


@pytest.fixture
def engine(tmp_path):
    be = FakeBackend()
    return T.TextToSpeech(T.TTSConfig(cache_dir=str(tmp_path)), backend=be), be


# ------------------------------------------------------------------ contract + caching
def test_speak_returns_audio_bytes(engine):
    e, be = engine
    audio = e.speak("What is your age?", "en")
    assert isinstance(audio, bytes) and audio.startswith(b"ID3")
    assert be.calls == [("What is your age?", "en", False)]


def test_second_call_comes_from_cache(engine):
    e, be = engine
    r1 = e.speak_full("What is your age?", "en")
    r2 = e.speak_full("What  is your age? ", "en")         # same text after cleaning
    assert not r1.cached and r2.cached
    assert r1.audio == r2.audio and len(be.calls) == 1
    assert e.cache_info()["files"] == 1


def test_cache_survives_a_new_instance_and_works_offline(tmp_path):
    T.TextToSpeech(T.TTSConfig(cache_dir=str(tmp_path)), FakeBackend()).speak("Is that correct?", "en")
    offline = T.TextToSpeech(T.TTSConfig(cache_dir=str(tmp_path)), FakeBackend(fail=True))
    assert offline.speak_full("Is that correct?", "en").cached
    with pytest.raises(T.TTSError):
        offline.speak("A new sentence", "en")


def test_cache_key_depends_on_language_and_speed(engine):
    e, be = engine
    e.speak("25", "en"); e.speak("25", "hi"); e.speak("25", "en", slow=True); e.speak("25", "en")
    assert len(be.calls) == 3


def test_broken_cache_file_is_replaced(engine):
    e, be = engine
    r = e.speak_full("hello", "en")
    with open(r.segments[0]["path"], "wb") as f:
        f.write(b"")                                         # e.g. an interrupted copy
    assert not e.speak_full("hello", "en").cached
    assert len(be.calls) == 2


def test_use_cache_false_never_writes(tmp_path):
    be = FakeBackend()
    e = T.TextToSpeech(T.TTSConfig(cache_dir=str(tmp_path), use_cache=False), be)
    e.speak("hello"); e.speak("hello")
    assert len(be.calls) == 2 and not any(tmp_path.iterdir())


def test_clear_cache(engine):
    e, _ = engine
    e.speak("one"); e.speak("two")
    assert e.clear_cache() == 2 and e.cache_info()["files"] == 0


def test_prefetch_reports_counts(engine):
    e, _ = engine
    out = e.prefetch(["How old are you?", "What is your PIN code?"], "en")
    assert out == {"spoken": 2, "already_cached": 0, "errors": []}
    assert e.prefetch(["How old are you?"], "en")["already_cached"] == 1


# ------------------------------------------------------------------ languages and scripts
@pytest.mark.parametrize("value,code", [(None, "en"), ("ML", "ml"), ("ml-IN", "ml"), ("Kannada", "kn"), ("hi_IN", "hi")])
def test_normalize_language(value, code):
    assert T.normalize_language(value) == code


def test_unsupported_language_is_an_error(engine):
    with pytest.raises(T.TTSError):
        engine[0].speak("hola", "es")


def test_split_by_script_mixed_sentence():
    parts = T.split_by_script("I heard your date of birth as 15 മാർച്ച് 2000. Is that correct?", "ml")
    assert parts == [("en", "I heard your date of birth as 15"), ("ml", "മാർച്ച് 2000."), ("en", "Is that correct?")]


@pytest.mark.parametrize("text,lang", [
    ("आपकी उम्र क्या है?", "hi"), ("ನಿಮ್ಮ ಹೆಸರು ಹೇಳಿ", "kn"), ("എന്റെ പേര്", "ml"), ("What is your name?", "en")])
def test_single_script_is_one_piece(text, lang):
    assert T.split_by_script(text, "en") == [(lang, text)]


def test_text_without_letters_uses_requested_language():
    assert T.split_by_script("9 8 4 7 2", "hi") == [("hi", "9 8 4 7 2")]


def test_malayalam_chillu_with_joiner_stays_in_one_piece():
    text = "അവന്‍ വന്നു"                                 # old-style chillu (n + virama + ZWJ)
    assert T.split_by_script(text, "ml") == [("ml", text)]


def test_mixed_sentence_uses_each_voice_and_joins_audio(engine):
    e, be = engine
    r = e.speak_full("Your blood group is ബി പോസിറ്റീവ്. Is that correct?", "ml")
    assert [c[1] for c in be.calls] == ["en", "ml", "en"]
    assert [s["language"] for s in r.segments] == ["en", "ml", "en"]
    assert r.audio == b"".join(FAKE_MP3 + f"{lang}:{t}".encode() for t, lang, _ in be.calls)


def test_one_voice_mode(engine):
    e, be = engine
    e.speak("Your age is ഇരുപത്", "ml", mixed_script=False)
    assert be.calls == [("Your age is ഇരുപത്", "ml", False)]


def test_empty_or_punctuation_only_text_is_an_error(engine):
    for t in ("", "   ", "?!"):
        with pytest.raises(T.TTSError):
            engine[0].speak(t)


def test_result_to_dict(engine):
    d = engine[0].speak_full("hello", "en").to_dict()
    assert d["language"] == "en" and d["format"] == "mp3" and d["bytes"] > 0 and "audio" not in d


def test_module_level_speak_uses_env_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SVARA_TTS_CACHE", str(tmp_path))
    T.reset_default()
    try:
        T.get_default().backend = FakeBackend()
        assert T.speak("hello", "en").startswith(b"ID3")
        assert T.cache_info()["dir"] == os.path.abspath(str(tmp_path))
    finally:
        T.reset_default()


def test_gtts_backend_error_is_wrapped(monkeypatch):
    gtts = pytest.importorskip("gtts")

    class Boom:
        def __init__(self, *a, **k):
            pass

        def write_to_fp(self, fp):
            raise gtts.tts.gTTSError("Failed to connect")

    monkeypatch.setattr(gtts, "gTTS", Boom)
    with pytest.raises(T.TTSError, match="no internet"):
        T.GTTSBackend().synthesize("hello", "en")


# ------------------------------------------------------------------ optional real request
@pytest.mark.skipif(os.environ.get("SVARA_ONLINE_TESTS") != "1", reason="set SVARA_ONLINE_TESTS=1 (needs internet)")
@pytest.mark.parametrize("text,lang", [("How old are you?", "en"), ("आपकी उम्र क्या है?", "hi"),
                                       ("നിങ്ങളുടെ പേര് പറയൂ", "ml"), ("ನಿಮ್ಮ ಹೆಸರು ಹೇಳಿ", "kn")])
def test_real_gtts_decodes_with_1a(tmp_path, text, lang):
    r = T.TextToSpeech(T.TTSConfig(cache_dir=str(tmp_path))).speak_full(text, lang)
    x, sr = r.to_signal()
    assert sr == 16000 and len(x) > 0.5 * sr
