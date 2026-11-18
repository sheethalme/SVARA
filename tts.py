"""
tts.py  --  SVARA, Member 2 / Sub-module 2C: Text-to-Speech with audio caching
==============================================================================

Public contract (used by the dialogue to ask questions and read answers back)::

    from tts import speak
    mp3 = speak("What is your date of birth?", "en")       # -> MP3 bytes
    st.audio(mp3, format="audio/mpeg", autoplay=True)        # play it in Streamlit

    speak("നിങ്ങളുടെ പേര് പറയൂ", "ml")                      # Malayalam voice
    speak_full(text, "hi").to_signal()                       # (float32 signal, 16000) via 1A

How it works
------------
* **Engine: gTTS** (Google Translate's text-to-speech).  Free, no API key, natural voices
  for all four project languages (en, hi, ml, kn).  English uses the Indian-English voice
  (``tld="co.in"``).  It needs internet for every *new* sentence.

* **Audio caching.**  Every synthesised piece is saved as an MP3 in a cache folder
  (default ``tts_cache/``, or ``SVARA_TTS_CACHE``).  The file name is a SHA-256 hash of
  (engine, language, accent, speed, text), so the same sentence in the same voice is never
  downloaded twice.  The fixed questions of a form ("What is your PIN code?", "Is that
  correct?") are therefore instant after the first user, and they keep working
  **offline**.  Files are written atomically (temporary file + rename), so an interrupted
  download never leaves a broken file; an empty or non-MP3 cache file is ignored and made
  again.

* **Mixed-script sentences use the right voice for each part.**  A read-back such as
  "I heard your date of birth as 15 മാർച്ച് 2000. Is that correct?" mixes English and
  Malayalam.  One Malayalam voice reads English letters badly, and an English voice cannot
  read Malayalam at all.  :func:`split_by_script` cuts the text where the script changes
  (Latin -> English voice, Malayalam / Kannada / Devanagari -> that language's voice),
  each piece is synthesised (and cached) separately, and the MP3s are joined.  Digits and
  punctuation stay with the piece they belong to.  Pass ``mixed_script=False`` to read the
  whole text with one voice.

* **Failures are explicit.**  No internet and not in the cache -> :class:`TTSError` with a
  plain message; the UI then falls back to the browser's own speech synthesis.

Privacy note: gTTS sends the *text* (not the user's voice) to Google.  Read-backs can
contain personal data (phone, Aadhaar); use ``engine`` / ``backend`` = an offline voice or
the browser fallback if that is not acceptable.

Command line::

    python tts.py "What is your age?" --lang en -o question.mp3
    python tts.py "നിങ്ങളുടെ പേര് പറയൂ" --lang ml -o name_ml.mp3
    python tts.py --cache-info | --clear-cache
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "LANGUAGES", "TTSError", "TTSConfig", "SpeechResult", "GTTSBackend", "TextToSpeech",
    "speak", "speak_full", "prefetch", "split_by_script", "normalize_language",
    "get_default", "reset_default", "clear_cache", "cache_info",
]

ROOT = os.path.dirname(os.path.abspath(__file__))
LANGUAGES: Dict[str, str] = {"en": "English", "hi": "Hindi", "ml": "Malayalam", "kn": "Kannada"}
_NAME_TO_CODE = {v.lower(): k for k, v in LANGUAGES.items()}
_SCRIPT_RANGES = (("hi", 0x0900, 0x097F), ("kn", 0x0C80, 0x0CFF), ("ml", 0x0D00, 0x0D7F))
MP3_MIME = "audio/mpeg"


class TTSError(RuntimeError):
    """Speech could not be produced (no internet and not cached, bad input, engine error)."""


# =========================================================================== config
@dataclass
class TTSConfig:
    cache_dir: str = field(default_factory=lambda: os.environ.get("SVARA_TTS_CACHE")
                           or os.path.join(ROOT, "tts_cache"))
    use_cache: bool = True
    tld: str = "co.in"            # Google domain = accent; co.in gives Indian English
    slow: bool = False            # gTTS "slow" voice (useful for spelling out letters)
    timeout_s: float = 10.0       # per request
    mixed_script: bool = True     # use each script's own voice (see module doc)


# =========================================================================== helpers
def normalize_language(value: Optional[str]) -> str:
    """'ml', 'ML', 'ml-IN', 'Malayalam' -> 'ml'.  ``None`` -> 'en'.  Anything else: TTSError."""
    if value is None or str(value).strip() == "":
        return "en"
    v = str(value).strip().lower().replace("_", "-")
    v = _NAME_TO_CODE.get(v, v.split("-")[0])
    if v not in LANGUAGES:
        raise TTSError(f"language {value!r} is not supported; use one of {', '.join(LANGUAGES)}")
    return v


def _char_language(ch: str) -> Optional[str]:
    """The voice a *letter* needs: 'en' for Latin letters, 'hi'/'kn'/'ml' for those scripts,
    ``None`` for digits, spaces, punctuation and anything else (neutral)."""
    o = ord(ch)
    for lang, lo, hi in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return lang
    if ch.isalpha() and "LATIN" in unicodedata.name(ch, ""):
        return "en"
    return None


def clean_text(text: str) -> str:
    """NFC, collapse white space, strip.  Zero-width joiners (needed by Malayalam chillu
    letters and some Kannada/Hindi conjuncts) are kept."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def split_by_script(text: str, language: str = "en") -> List[Tuple[str, str]]:
    """Cut ``text`` where the script changes and give each piece its voice language.

    ``"I heard 15 മാർച്ച് 2000. Is that correct?"`` ->
    ``[("en", "I heard 15"), ("ml", "മാർച്ച് 2000."), ("en", "Is that correct?")]``

    Neutral characters (digits, spaces, punctuation) stay with the piece before them; text
    with no letters at all (e.g. "9847216530") is read in ``language``.  Joiner and vowel
    sign characters (``Mn``/``Mc``/``Cf``) belong to the letter before them.
    """
    language = normalize_language(language)
    text = clean_text(text)
    if not text:
        return []
    pieces: List[List[str]] = []          # [lang, text]
    pending = ""                          # neutral text seen before the first letter
    for ch in text:
        lang = _char_language(ch)
        if lang is None or (pieces and pieces[-1][0] == lang):
            if pieces:
                pieces[-1][1] += ch
            else:
                pending += ch
            continue
        pieces.append([lang, pending + ch])   # the script changed: start a new piece
        pending = ""
    if not pieces:
        return [(language, text)]
    return [(lang, t.strip()) for lang, t in pieces if t.strip()]


def _looks_like_mp3(data: bytes) -> bool:
    return len(data) > 32 and (data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0))


# =========================================================================== engine
class GTTSBackend:
    """gTTS (Google Translate text-to-speech).  ``synthesize`` returns MP3 bytes."""

    name = "gtts"

    def __init__(self, tld: str = "co.in", timeout_s: float = 10.0):
        self.tld, self.timeout_s = tld, timeout_s

    @property
    def available(self) -> bool:
        try:
            import gtts  # noqa: F401
            return True
        except ImportError:
            return False

    def cache_tag(self, language: str, slow: bool) -> str:
        return f"{self.name}|{language}|{self.tld}|{'slow' if slow else 'normal'}"

    def synthesize(self, text: str, language: str, slow: bool = False) -> bytes:
        try:
            from gtts import gTTS
            from gtts.tts import gTTSError
        except ImportError as e:
            raise TTSError("gTTS is not installed: pip install gTTS") from e
        buf = io.BytesIO()
        try:
            gTTS(text=text, lang=language, tld=self.tld, slow=slow, timeout=self.timeout_s).write_to_fp(buf)
        except gTTSError as e:
            raise TTSError(f"gTTS could not synthesise the text (no internet, or Google refused): {e}") from e
        except Exception as e:  # requests / socket errors surface as many types
            raise TTSError(f"gTTS failed: {type(e).__name__}: {e}") from e
        data = buf.getvalue()
        if not _looks_like_mp3(data):
            raise TTSError("gTTS returned no audio")
        return data


# =========================================================================== result
@dataclass
class SpeechResult:
    audio: bytes                                   # MP3 bytes (play, save, or decode)
    text: str                                      # the text that was spoken (cleaned)
    language: str                                  # the requested language
    segments: List[Dict[str, Any]] = field(default_factory=list)   # [{language, text, cached, path}]
    engine: str = "gtts"
    latency_s: float = 0.0
    format: str = "mp3"
    mime: str = MP3_MIME

    @property
    def cached(self) -> bool:
        """True when every piece came from the cache (no network was used)."""
        return bool(self.segments) and all(s["cached"] for s in self.segments)

    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            f.write(self.audio)
        return path

    def to_signal(self, sample_rate: Optional[int] = None):
        """Decode with sub-module 1A: ``(float32 mono signal, 16000)`` -- e.g. to feed TTS
        audio into the recogniser for tests."""
        from audio_io import load_audio
        return load_audio(self.audio, sample_rate) if sample_rate else load_audio(self.audio)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "language": self.language, "engine": self.engine,
                "cached": self.cached, "latency_s": round(self.latency_s, 3), "bytes": len(self.audio),
                "format": self.format, "segments": self.segments}


# =========================================================================== main class
class TextToSpeech:
    """Text -> speech with a disk cache.  ``backend`` is anything with
    ``synthesize(text, language, slow) -> bytes`` and ``cache_tag(language, slow) -> str``
    (gTTS by default; tests pass a fake one)."""

    def __init__(self, config: Optional[TTSConfig] = None, backend: Any = None):
        self.cfg = config or TTSConfig()
        self.backend = backend or GTTSBackend(self.cfg.tld, self.cfg.timeout_s)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cache
    def _cache_path(self, text: str, language: str, slow: bool) -> str:
        key = hashlib.sha256(f"{self.backend.cache_tag(language, slow)}|{text}".encode("utf-8")).hexdigest()
        return os.path.join(self.cfg.cache_dir, key[:2], key + ".mp3")

    def _read_cache(self, path: str) -> Optional[bytes]:
        if not self.cfg.use_cache or not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return None
        if not _looks_like_mp3(data):
            log.warning("ignoring broken TTS cache file %s", path)
            return None
        return data

    def _write_cache(self, path: str, data: bytes) -> None:
        if not self.cfg.use_cache:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except OSError as e:                      # a read-only disk must not break speech
            log.warning("could not write TTS cache %s: %s", path, e)

    # ------------------------------------------------------------------ synthesis
    def _piece(self, text: str, language: str, slow: bool) -> Tuple[bytes, bool, str]:
        path = self._cache_path(text, language, slow)
        data = self._read_cache(path)
        if data is not None:
            return data, True, path
        with self._lock:                          # one download per sentence, even with reruns
            data = self._read_cache(path)
            if data is not None:
                return data, True, path
            data = self.backend.synthesize(text, language, slow)
            self._write_cache(path, data)
        return data, False, path

    def speak_full(self, text: str, language: Optional[str] = "en", slow: Optional[bool] = None,
                   mixed_script: Optional[bool] = None) -> SpeechResult:
        t0 = time.perf_counter()
        language = normalize_language(language)
        text = clean_text(text)
        if not text:
            raise TTSError("nothing to say: the text is empty")
        slow = self.cfg.slow if slow is None else slow
        mixed = self.cfg.mixed_script if mixed_script is None else mixed_script
        parts = split_by_script(text, language) if mixed else [(language, text)]
        audio, segments = [], []
        for lang, piece in parts:
            if not any(c.isalnum() for c in piece):   # a lone "." or "?" would be silence or an error
                continue
            data, hit, path = self._piece(piece, lang, slow)
            audio.append(data)
            segments.append({"language": lang, "text": piece, "cached": hit, "path": path})
        if not audio:
            raise TTSError("nothing to say: the text has no letters or digits")
        return SpeechResult(audio=b"".join(audio), text=text, language=language, segments=segments,
                            engine=getattr(self.backend, "name", "custom"),
                            latency_s=time.perf_counter() - t0)

    def speak(self, text: str, language: Optional[str] = "en", **kw: Any) -> bytes:
        return self.speak_full(text, language, **kw).audio

    def prefetch(self, texts: Iterable[str], language: Optional[str] = "en") -> Dict[str, Any]:
        """Synthesise ahead of time (e.g. all questions of a form when the admin saves it),
        so users get instant, offline-safe audio.  Returns counts and the errors seen."""
        done = from_cache = 0
        errors: List[str] = []
        for t in texts:
            try:
                r = self.speak_full(t, language)
                done += 1
                from_cache += r.cached
            except TTSError as e:
                errors.append(f"{t[:40]!r}: {e}")
        return {"spoken": done, "already_cached": from_cache, "errors": errors}

    # ------------------------------------------------------------------ cache admin
    def cache_info(self) -> Dict[str, Any]:
        n = size = 0
        for dirpath, _, files in os.walk(self.cfg.cache_dir):
            for f in files:
                if f.endswith(".mp3"):
                    n += 1
                    size += os.path.getsize(os.path.join(dirpath, f))
        return {"dir": os.path.abspath(self.cfg.cache_dir), "files": n, "bytes": size}

    def clear_cache(self) -> int:
        removed = 0
        for dirpath, _, files in os.walk(self.cfg.cache_dir, topdown=False):
            for f in files:
                if f.endswith((".mp3", ".part")):
                    try:
                        os.remove(os.path.join(dirpath, f))
                        removed += 1
                    except OSError:
                        pass
            if dirpath != self.cfg.cache_dir:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
        return removed


# =========================================================================== module-level API
_DEFAULT: List[TextToSpeech] = []
_DEFAULT_LOCK = threading.Lock()


def get_default() -> TextToSpeech:
    with _DEFAULT_LOCK:
        if not _DEFAULT:
            _DEFAULT.append(TextToSpeech())
        return _DEFAULT[0]


def reset_default() -> None:
    """Forget the shared instance (e.g. after changing SVARA_TTS_CACHE)."""
    with _DEFAULT_LOCK:
        _DEFAULT.clear()


def speak(text: str, language: Optional[str] = "en", **kw: Any) -> bytes:
    """The 2C contract: ``text`` in ``language`` (en / hi / ml / kn) -> MP3 audio bytes."""
    return get_default().speak(text, language, **kw)


def speak_full(text: str, language: Optional[str] = "en", **kw: Any) -> SpeechResult:
    """Like :func:`speak` but returns a :class:`SpeechResult` (cache hits, pieces, timing)."""
    return get_default().speak_full(text, language, **kw)


def prefetch(texts: Iterable[str], language: Optional[str] = "en") -> Dict[str, Any]:
    return get_default().prefetch(texts, language)


def cache_info() -> Dict[str, Any]:
    return get_default().cache_info()


def clear_cache() -> int:
    return get_default().clear_cache()


# =========================================================================== CLI
def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse
    import json
    import sys
    if hasattr(sys.stdout, "reconfigure"):          # Windows consoles: print Malayalam etc.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="SVARA 2C: text-to-speech (gTTS) with caching.")
    p.add_argument("text", nargs="?")
    p.add_argument("--lang", default="en", help="en / hi / ml / kn")
    p.add_argument("-o", "--out", help="write the MP3 here (default: tts_out.mp3)")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--one-voice", action="store_true", help="do not switch voice for mixed-script text")
    p.add_argument("--cache-info", action="store_true")
    p.add_argument("--clear-cache", action="store_true")
    a = p.parse_args(argv)
    if a.clear_cache:
        print(f"removed {clear_cache()} cached file(s)")
        return 0
    if a.cache_info:
        print(json.dumps(cache_info(), indent=2))
        return 0
    if not a.text:
        p.error("give the text to speak")
    try:
        r = speak_full(a.text, a.lang, slow=a.slow, mixed_script=not a.one_voice)
    except TTSError as e:
        print("ERROR:", e)
        return 1
    out = r.save(a.out or "tts_out.mp3")
    print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    print("written:", os.path.abspath(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
