"""Helpers for tests and demos (NOT used in a real call)."""
from __future__ import annotations

from typing import List

from callpage.session import TranscribeResult, Transcriber


def scripted_transcriber(texts: List[str]) -> Transcriber:
    """Returns the given transcripts one by one, ignoring the audio (the last one repeats)."""
    state = {"i": 0}

    def transcribe(wav: bytes, language: str, hint=None) -> TranscribeResult:
        i = min(state["i"], len(texts) - 1)
        state["i"] += 1
        return TranscribeResult(text=texts[i], reliable=True, language=language, engine="scripted")

    return transcribe
