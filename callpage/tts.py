"""
tts.py -- text-to-speech for the call (server side, gTTS), with a disk cache
==============================================================================

``synth(text, lang)`` returns MP3 bytes.  It needs the optional ``gTTS`` package (``pip install gTTS``) and an
internet connection (gTTS calls Google's online voice).  If either is missing it raises ``TTSUnavailable`` and the
page falls back to the browser's built-in voice (``speechSynthesis``), which works offline but may not have
Malayalam / Kannada voices installed.

The same sentences are asked again and again, so results are cached in ``data/tts_cache`` (keyed by language + text).
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional

LANG_CODE = {"en": "en", "hi": "hi", "ml": "ml", "kn": "kn"}
TLD = {"en": "co.in"}                       # Indian English accent
MAX_CHARS = 700
_lock = threading.Lock()


class TTSUnavailable(RuntimeError):
    pass


def synth(text: str, lang: str, cache_dir: Optional[str] = None) -> bytes:
    text = (text or "").strip()
    if not text:
        raise TTSUnavailable("nothing to say")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    code = LANG_CODE.get(lang, "en")
    path = None
    if cache_dir:
        key = hashlib.sha1(f"{code}|{text}".encode("utf-8")).hexdigest()
        path = os.path.join(cache_dir, f"{code}_{key}.mp3")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
    try:
        from gtts import gTTS
    except ImportError:
        raise TTSUnavailable("the gTTS package is not installed (pip install gTTS)") from None
    try:
        import io
        buf = io.BytesIO()
        with _lock:
            gTTS(text=text, lang=code, tld=TLD.get(code, "com")).write_to_fp(buf)
        data = buf.getvalue()
    except Exception as exc:                          # noqa: BLE001 - network, blocked host, rate limit ...
        raise TTSUnavailable(f"online voice failed: {type(exc).__name__}") from None
    if not data:
        raise TTSUnavailable("online voice returned nothing")
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    return data
