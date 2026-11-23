"""
pipeline.py -- the REAL speech pipeline used by the call:  WAV bytes -> 1A load -> (1B clean) -> 2A speech-to-text
=================================================================================================================

``make_transcriber(preprocess="off", engine="auto")`` returns the function the call controller needs.

preprocess:
  "off"   (default) only 1A (mono, 16 kHz, level).  In my own tests the 1B band-pass made Whisper WORSE on clean
          speech, and the browser already trims silence and cancels echo / noise, so the safe default is off.
  "light" 1B noise reduction + silence trimming, without the band-pass and pre-emphasis.
  "full"  the whole 1B chain.
Set it with ``SVARA_CALL_PREPROCESS`` or ``--preprocess`` and compare on your own recordings.

engine: "auto" (cloud first, local fallback), "cloud" or "local"  (``SVARA_CALL_ENGINE``).
"""

from __future__ import annotations

import os
from typing import Optional

from callpage.session import TranscribeResult, Transcriber


def make_transcriber(preprocess: Optional[str] = None, engine: Optional[str] = None) -> Transcriber:
    preprocess = (preprocess or os.environ.get("SVARA_CALL_PREPROCESS") or "off").lower()
    engine = (engine or os.environ.get("SVARA_CALL_ENGINE") or "auto").lower()
    if preprocess not in ("off", "light", "full"):
        raise ValueError("preprocess must be off, light or full")

    def transcribe(wav: bytes, language: str, hint: Optional[str] = None) -> TranscribeResult:
        import audio_io
        import stt
        try:
            x, sr = audio_io.load_audio(wav)                          # 1A
        except audio_io.AudioLoadError as exc:
            return TranscribeResult(error=f"unreadable audio: {exc}")
        if preprocess != "off":
            import preprocess as pp                                   # 1B
            kw = {"pre_emphasis": None, "bandpass": False} if preprocess == "light" else {}
            x = pp.clean_audio(x, sr, **kw)
        try:
            r = stt.transcribe_full(x, language=language, prompt=hint, engine=engine)   # 2A
        except (stt.STTError, ValueError) as exc:
            return TranscribeResult(error=str(exc)[:200])
        return TranscribeResult(text=r.text, reliable=r.reliable, language=r.language, engine=r.engine,
                                warnings=list(r.warnings), skipped=r.skipped)

    return transcribe
