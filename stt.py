"""
stt.py  --  SVARA, Member 2 / Sub-module 2A: Speech-to-Text and Language ID
===========================================================================

Public contract (consumed by 2B ``extract_field`` and by Member 3's dialogue)::

    text, language = transcribe(audio)       # language is "ml" | "kn" | "hi" | "en" | "unknown"

``audio`` is what sub-module 1B returned (``clean_audio`` -> 1-D float32, 16 kHz), or
anything sub-module 1A accepts (file path, bytes, upload, ``(signal, sr)``).

How it works
------------
* **Two engines with automatic failover**

  1. *Cloud* (primary): any OpenAI-compatible ``/audio/transcriptions`` endpoint -- Groq
     (``whisper-large-v3``) or OpenAI (``whisper-1``).  Chosen by environment variables
     (``GROQ_API_KEY`` or ``OPENAI_API_KEY``).
  2. *Local* (fallback): Whisper ``small`` running on this machine through ``faster-whisper``
     (CTranslate2, int8).  Works offline.

  Order: cloud first, local if the cloud is missing/slow/rejects the request.  A small
  *circuit breaker* skips the cloud for 60 s after repeated failures, so an offline demo is
  not slowed by a network time-out on every answer.

* **Language identification restricted to the four project languages.**  Whisper scores
  ~99 languages; confusable neighbours (Tamil/Malayalam, Telugu/Kannada, Urdu/Hindi) would
  otherwise be picked now and then.  The local engine takes Whisper's own probabilities and
  keeps only ml/kn/hi/en (renormalised); the cloud result is cross-checked against the
  *script* of the text (Malayalam, Kannada, Devanagari or Latin letters).

* **Safeguards:** silence/too-short audio never reaches an engine (Whisper invents text on
  silence); repetition loops are dropped; every decision is explained in the result's
  ``warnings`` and ``fallback_reason``.

Privacy note: the cloud engine uploads the recording to a third party.  Use
``engine="local"`` (or no API key) for fully offline processing of personal data.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

from audio_io import TARGET_SR, AudioLoadError, load_audio

log = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_LANGUAGES", "UNKNOWN", "STTConfig", "STTError", "TranscriptionResult",
    "SpeechToText", "CloudEngine", "LocalEngine",
    "transcribe", "transcribe_full", "detect_language", "get_default", "reset_default",
    "language_from_script", "restrict_language_probs", "normalize_language",
]

# =========================================================================== constants
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "ml": "Malayalam", "kn": "Kannada", "hi": "Hindi", "en": "English"}
UNKNOWN = "unknown"

# Whisper's names for languages -> ISO codes (only those we may meet; unknown names pass through)
_NAME_TO_CODE = {
    "malayalam": "ml", "kannada": "kn", "hindi": "hi", "english": "en",
    "tamil": "ta", "telugu": "te", "urdu": "ur", "marathi": "mr", "nepali": "ne",
    "punjabi": "pa", "gujarati": "gu", "bengali": "bn", "sanskrit": "sa", "sindhi": "sd",
    "assamese": "as", "sinhala": "si", "arabic": "ar", "persian": "fa", "spanish": "es",
    "french": "fr", "german": "de", "chinese": "zh", "japanese": "ja", "russian": "ru",
}
# When Whisper names a language outside the four, the most likely intended neighbour.
_CONFUSION = {"ta": "ml", "te": "kn", "ur": "hi", "mr": "hi", "ne": "hi", "pa": "hi",
              "gu": "hi", "bn": "hi", "sa": "hi", "sd": "hi", "as": "hi"}
_SCRIPT_RANGES = {"hi": (0x0900, 0x097F), "kn": (0x0C80, 0x0CFF), "ml": (0x0D00, 0x0D7F)}

# Local models too small to handle Malayalam / Kannada (measured: no usable output, see docs)
_WEAK_LOCAL_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en"}
_WEAK_LANGUAGES = ("ml", "kn")

_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "whisper-large-v3", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "whisper-1", "OPENAI_API_KEY"),
}


class STTError(RuntimeError):
    """Raised when no engine could transcribe the audio."""


class CloudError(RuntimeError):
    """``skip=True`` means the cloud was not even tried (no key / breaker open)."""

    def __init__(self, msg: str, retryable: bool = False, status: Optional[int] = None, skip: bool = False):
        super().__init__(msg)
        self.retryable, self.status, self.skip = retryable, status, skip


class LocalEngineError(RuntimeError):
    pass


# =========================================================================== config
@dataclass
class STTConfig:
    """All settings.  ``STTConfig.from_env()`` fills the secrets from environment variables."""

    engine: str = "auto"                       # "auto" (cloud, then local) | "cloud" | "local"
    # --- cloud
    cloud_provider: str = "auto"               # "auto" | "groq" | "openai" | "custom"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cloud_model: Optional[str] = None
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 30.0
    cloud_retries: int = 1                     # extra attempts for time-outs / 5xx / 429
    retry_backoff_s: float = 0.5
    breaker_failures: int = 2                  # consecutive failures that open the breaker
    breaker_cooldown_s: float = 60.0
    max_upload_bytes: int = 24_000_000         # providers limit uploads to ~25 MB
    # --- local
    local_model: str = "small"                 # name, or folder of a converted model (offline)
    specialists: Dict[str, str] = field(default_factory=dict)
    # language -> local model used for that language instead, e.g. {"ml": "<Malayalam fine-tuned CTranslate2 repo>"}.
    # Stock Whisper cannot transcribe Malayalam (measured); a fine-tuned model can.
    local_device: str = "auto"
    local_compute_type: str = "auto"           # "auto" = int8 on CPU, float16 on GPU
    local_files_only: bool = False             # True: never touch the network for the model
    beam_size: int = 5
    local_temperature_fallback: bool = False   # Whisper's retry-at-higher-temperature loop: slower (often 5-10x
                                               # on Malayalam/Kannada) and rarely helps short answers
    cpu_threads: int = 0                       # 0 = library default
    # --- behaviour
    min_audio_s: float = 0.2
    silence_peak: float = 1e-3
    low_language_confidence: float = 0.6
    supported: Tuple[str, ...] = tuple(SUPPORTED_LANGUAGES)

    @classmethod
    def from_env(cls, **overrides: Any) -> "STTConfig":
        try:                                    # optional: read a .env file if python-dotenv is present
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:                       # noqa: BLE001
            pass
        env = os.environ
        cfg = cls(
            engine=env.get("SVARA_STT_ENGINE", "auto"),
            cloud_provider=env.get("SVARA_STT_PROVIDER", "auto"),
            base_url=env.get("SVARA_STT_BASE_URL") or None,
            cloud_model=env.get("SVARA_STT_MODEL") or None,
            local_model=env.get("SVARA_WHISPER_PATH") or env.get("SVARA_WHISPER_MODEL", "small"),
            local_files_only=env.get("SVARA_STT_OFFLINE", "0") == "1",
            specialists={c: env[f"SVARA_WHISPER_MODEL_{c.upper()}"] for c in SUPPORTED_LANGUAGES
                         if env.get(f"SVARA_WHISPER_MODEL_{c.upper()}")},
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown STTConfig field {k!r}")
            setattr(cfg, k, v)
        return cfg


# =========================================================================== language helpers
def normalize_language(value: Optional[str]) -> Optional[str]:
    """'Malayalam' / 'malayalam' / 'ml' / 'ml-IN' -> 'ml'.  Unknown names pass through lower-cased."""
    if value is None:
        return None
    v = str(value).strip().lower().replace("_", "-")
    if not v:
        return None
    if v in _NAME_TO_CODE:
        return _NAME_TO_CODE[v]
    base = v.split("-")[0]
    return base


def script_fractions(text: str) -> Dict[str, float]:
    """Share of the *letters* in ``text`` written in Devanagari (hi), Kannada (kn),
    Malayalam (ml) or Latin (en).  Digits, punctuation and combining marks are ignored."""
    counts = {"hi": 0, "kn": 0, "ml": 0, "en": 0}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        for code, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[code] += 1
                break
        else:
            if "LATIN" in unicodedata.name(ch, ""):
                counts["en"] += 1
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def language_from_script(text: str, min_letters: int = 3, threshold: float = 0.7) -> Optional[str]:
    """Language implied by the writing system, or None if mixed / too short."""
    letters = sum(1 for c in text if c.isalpha())
    if letters < min_letters:
        return None
    fr = script_fractions(text)
    best = max(fr, key=fr.get)
    return best if fr[best] >= threshold else None


def restrict_language_probs(all_probs: Sequence[Tuple[str, float]],
                            supported: Sequence[str] = tuple(SUPPORTED_LANGUAGES)
                            ) -> Tuple[str, float, Dict[str, float], float]:
    """Keep only the supported languages and renormalise.

    Returns ``(best, confidence, probs, supported_mass)`` where ``probs`` sums to 1 over the
    supported languages and ``supported_mass`` is the share of Whisper's belief that was on
    them (a small value means the speech is probably in some other language)."""
    raw = {code: 0.0 for code in supported}
    for code, p in all_probs:
        c = normalize_language(code)
        if c in raw:
            raw[c] += float(p)
    mass = sum(raw.values())
    if mass <= 0:
        return supported[-1] if "en" in supported else supported[0], 0.0, {c: 0.0 for c in supported}, 0.0
    probs = {c: v / mass for c, v in raw.items()}
    best = max(probs, key=probs.get)
    return best, probs[best], probs, mass


def _clean_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", t).strip()


def _is_repetition_loop(text: str) -> bool:
    """Whisper sometimes degenerates into repeating one phrase; detect and discard it."""
    toks = text.split()
    if len(toks) >= 8 and len(set(toks)) / len(toks) < 0.25:
        return True
    return bool(re.search(r"(.{2,12}?)\1{6,}", text))


# =========================================================================== results
@dataclass
class EngineResult:
    text: str
    language: str                              # ISO code if known, else Whisper's name lower-cased
    engine: str
    model: str
    language_probs: Optional[Dict[str, float]] = None
    language_confidence: Optional[float] = None
    supported_mass: Optional[float] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    latency_s: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    engine: str = ""                           # "cloud" | "local" | "" (nothing ran)
    model: str = ""
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    language_confidence: Optional[float] = None
    language_probs: Optional[Dict[str, float]] = None
    audio_seconds: float = 0.0
    latency_s: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped: Optional[str] = None              # why no engine was called (e.g. "silence")
    reliable: bool = True                      # False => ask the user to confirm / repeat (see warnings)

    def as_tuple(self) -> Tuple[str, str]:
        return self.text, self.language

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================== cloud engine
def _to_wav_bytes(x: np.ndarray, sr: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class CloudEngine:
    """OpenAI-compatible ``POST {base_url}/audio/transcriptions`` client (Groq, OpenAI, ...)."""

    name = "cloud"

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self.provider = None
        self.base_url = cfg.base_url
        self.model = cfg.cloud_model
        self.api_key = cfg.api_key
        prov = (cfg.cloud_provider or "auto").lower()
        if prov == "auto":
            if cfg.base_url:
                prov = "custom"
            elif self.api_key or os.environ.get("GROQ_API_KEY"):
                prov = "groq"
            elif os.environ.get("OPENAI_API_KEY"):
                prov = "openai"
            else:
                prov = "none"
        self.provider = prov
        if prov in _PROVIDERS:
            base, model, env_key = _PROVIDERS[prov]
            self.base_url = self.base_url or base
            self.model = self.model or model
            self.api_key = self.api_key or os.environ.get(env_key)
        elif prov == "custom":
            self.api_key = self.api_key or os.environ.get("SVARA_STT_API_KEY")
            self.model = self.model or "whisper-1"
        if self.base_url:
            u = urlparse(self.base_url)
            local = u.hostname in ("localhost", "127.0.0.1", "::1")
            if u.scheme != "https" and not local:
                raise ValueError("base_url must use https (the API key would be sent in clear text)")
        self._session = None

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url and self.provider != "none")

    @property
    def model_name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _post(self, wav: bytes, language: Optional[str], prompt: Optional[str]):
        import requests
        if self._session is None:
            self._session = requests.Session()
        data = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        return self._session.post(
            self.base_url.rstrip("/") + "/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            data=data, files={"file": ("audio.wav", wav, "audio/wav")},
            timeout=(self.cfg.connect_timeout_s, self.cfg.read_timeout_s))

    def _scrub(self, s: str) -> str:
        s = (s or "")[:200]
        return s.replace(self.api_key, "***") if self.api_key else s

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   prompt: Optional[str] = None) -> EngineResult:
        import requests
        if not self.available:
            raise CloudError("no cloud API key configured")
        wav = _to_wav_bytes(audio)
        if len(wav) > self.cfg.max_upload_bytes:
            raise CloudError(f"audio too large for the cloud API ({len(wav) / 1e6:.1f} MB)")
        t0 = time.perf_counter()
        last: Optional[CloudError] = None
        for attempt in range(self.cfg.cloud_retries + 1):
            if attempt:
                time.sleep(self.cfg.retry_backoff_s * (2 ** (attempt - 1)))
            try:
                r = self._post(wav, language, prompt)
            except requests.exceptions.Timeout:
                last = CloudError("cloud request timed out", retryable=True)
                continue
            except requests.exceptions.RequestException as exc:
                last = CloudError(f"cloud network error: {type(exc).__name__}", retryable=True)
                continue
            if r.status_code == 200:
                return self._parse(r, time.perf_counter() - t0, len(audio) / TARGET_SR)
            body = self._scrub(r.text)
            if r.status_code in (401, 403):
                raise CloudError(f"cloud rejected the API key (HTTP {r.status_code})", status=r.status_code)
            if r.status_code == 429 or r.status_code >= 500:
                last = CloudError(f"cloud busy/unavailable (HTTP {r.status_code})", True, r.status_code)
                continue
            raise CloudError(f"cloud error HTTP {r.status_code}: {body}", status=r.status_code)
        assert last is not None
        raise last

    def _parse(self, r, latency: float, duration: float) -> EngineResult:
        try:
            js = r.json()
            text = js["text"]
            if not isinstance(text, str):
                raise TypeError("text is not a string")
        except Exception as exc:                # noqa: BLE001
            raise CloudError(f"unreadable cloud response ({type(exc).__name__})", retryable=False) from None
        lang = normalize_language(js.get("language")) or UNKNOWN
        segs = []
        for s in js.get("segments") or []:
            segs.append({"start": s.get("start"), "end": s.get("end"), "text": s.get("text", ""),
                         "avg_logprob": s.get("avg_logprob"), "no_speech_prob": s.get("no_speech_prob")})
        return EngineResult(text=text, language=lang, engine="cloud", model=self.model_name,
                            segments=segs, duration_s=float(js.get("duration") or duration), latency_s=latency)


# =========================================================================== local engine
class LocalEngine:
    """Whisper ``small`` on this machine via faster-whisper (CTranslate2)."""

    name = "local"

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def model_id_for(self, language: Optional[str]) -> str:
        """Local model used for ``language`` (a specialist if configured, else the default)."""
        return self.cfg.specialists.get(language or "", "") or self.cfg.local_model

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def model_name(self) -> str:
        return f"faster-whisper:{self.cfg.local_model}"

    def _load(self, model_id: Optional[str] = None):
        model_id = model_id or self.cfg.local_model
        if model_id in self._models:
            return self._models[model_id]
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise LocalEngineError("the local engine needs faster-whisper: pip install faster-whisper") from None
        device, compute = self.cfg.local_device, self.cfg.local_compute_type
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:                   # noqa: BLE001
                device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        log.info("loading Whisper %s on %s (%s)", model_id, device, compute)
        try:
            self._models[model_id] = WhisperModel(model_id, device=device, compute_type=compute,
                                                  cpu_threads=self.cfg.cpu_threads,
                                                  local_files_only=self.cfg.local_files_only)
        except Exception as exc:                # noqa: BLE001
            raise LocalEngineError(
                f"could not load the Whisper '{model_id}' model ({type(exc).__name__}: "
                f"{str(exc)[:150]}). Download it once while online with:  python stt.py --download"
            ) from None
        return self._models[model_id]

    def preload(self, include_specialists: bool = True) -> None:
        with self._lock:
            self._load()
            if include_specialists:
                for mid in dict.fromkeys(self.cfg.specialists.values()):
                    self._load(mid)

    def detect_language(self, audio: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        with self._lock:
            model = self._load()                # language ID always uses the default (multilingual) model
            _, _, probs = model.detect_language(audio=np.asarray(audio, dtype=np.float32))
        return restrict_language_probs(probs, self.cfg.supported)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                   prompt: Optional[str] = None) -> EngineResult:
        t0 = time.perf_counter()
        x = np.asarray(audio, dtype=np.float32)
        notes: List[str] = []
        with self._lock:
            default_id = self.cfg.local_model
            probs = conf = mass = None
            if language is not None:
                used = self.model_id_for(language)
                segs, info = self._run(self._load(used), x, language, prompt)
                lang = language
            else:
                used = default_id
                model = self._load(default_id)
                segs, info = self._run(model, x, None, prompt)
                lang = normalize_language(info.language) or UNKNOWN
                if info.all_language_probs:
                    best, conf, probs, mass = restrict_language_probs(info.all_language_probs, self.cfg.supported)
                    if best != lang:           # Whisper preferred a language outside our four (or a different one)
                        notes.append(f"Whisper first chose '{lang}'; restricted to the supported languages -> '{best}'")
                        lang = best
                        segs, info = self._run(model, x, lang, prompt)
                    elif lang not in self.cfg.supported:
                        lang = best
                spec = self.model_id_for(lang)
                if spec != default_id:         # a specialist model exists for the detected language
                    notes.append(f"Using the specialist model '{spec}' for '{lang}'")
                    used = spec
                    segs, info = self._run(self._load(spec), x, lang, prompt)
        out_segments = [{"start": s.start, "end": s.end, "text": s.text,
                         "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob} for s in segs]
        text = " ".join(s.text.strip() for s in segs)
        return EngineResult(text=text, language=lang or UNKNOWN, engine="local", model=f"faster-whisper:{used}",
                            language_probs=probs, language_confidence=conf, supported_mass=mass,
                            segments=out_segments, duration_s=float(info.duration),
                            latency_s=time.perf_counter() - t0, notes=notes)

    def _run(self, model, x, language, prompt):
        kw: Dict[str, Any] = {}
        if not self.cfg.local_temperature_fallback:
            kw["temperature"] = 0.0
        gen, info = model.transcribe(
            x, language=language, task="transcribe", beam_size=self.cfg.beam_size,
            condition_on_previous_text=False, initial_prompt=prompt, vad_filter=False, **kw)
        return list(gen), info


# =========================================================================== orchestrator
class SpeechToText:
    """Cloud-first transcription with automatic local fallback.  See the module docstring."""

    def __init__(self, config: Optional[STTConfig] = None, cloud: Any = None, local: Any = None):
        self.cfg = config or STTConfig.from_env()
        self.cloud = cloud if cloud is not None else CloudEngine(self.cfg)
        self.local = local if local is not None else LocalEngine(self.cfg)
        self._cloud_failures = 0
        self._cloud_open_until = 0.0
        self._lock = threading.Lock()

    # ---- circuit breaker ------------------------------------------------------------
    def _cloud_ready(self) -> Tuple[bool, str]:
        if not getattr(self.cloud, "available", False):
            return False, "no cloud API key configured"
        if time.monotonic() < self._cloud_open_until:
            left = self._cloud_open_until - time.monotonic()
            return False, f"cloud skipped after repeated failures (retry in {left:.0f} s)"
        return True, ""

    def _cloud_ok(self) -> None:
        with self._lock:
            self._cloud_failures, self._cloud_open_until = 0, 0.0

    def _cloud_failed(self) -> None:
        with self._lock:
            self._cloud_failures += 1
            if self._cloud_failures >= self.cfg.breaker_failures:
                self._cloud_open_until = time.monotonic() + self.cfg.breaker_cooldown_s

    def warm_up(self) -> None:
        """Load the local model now (takes seconds) so the first answer is not delayed."""
        if hasattr(self.local, "preload"):
            self.local.preload()

    # ---- input ----------------------------------------------------------------------
    def _prepare(self, source: Any, sample_rate: Optional[int], preprocess: bool) -> np.ndarray:
        if preprocess:
            from preprocess import clean_audio
            return clean_audio(source, sample_rate)
        if isinstance(source, np.ndarray) and source.ndim == 1 and source.dtype.kind == "f" \
                and sample_rate in (None, TARGET_SR):
            if not np.all(np.isfinite(source)):
                raise AudioLoadError("The audio contains NaN or infinite values")
            return source.astype(np.float32, copy=False)      # already 1B output: leave untouched
        if isinstance(source, np.ndarray) and sample_rate is None:
            sample_rate = TARGET_SR
        x, _ = load_audio(source, sample_rate)
        return x

    # ---- main entry -----------------------------------------------------------------
    def transcribe_full(self, source: Any, sample_rate: Optional[int] = None, *,
                        language: Optional[str] = None, prompt: Optional[str] = None,
                        engine: Optional[str] = None, preprocess: bool = False) -> TranscriptionResult:
        hint = normalize_language(language)
        if hint is not None and hint not in self.cfg.supported:
            raise ValueError(f"language {language!r} is not supported; use one of {list(self.cfg.supported)}")
        mode = (engine or self.cfg.engine or "auto").lower()
        if mode not in ("auto", "cloud", "local"):
            raise ValueError("engine must be 'auto', 'cloud' or 'local'")

        x = self._prepare(source, sample_rate, preprocess)
        res = TranscriptionResult(text="", language=hint or UNKNOWN, audio_seconds=len(x) / TARGET_SR)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if res.audio_seconds < self.cfg.min_audio_s:
            res.skipped = "too short"
        elif peak < self.cfg.silence_peak:
            res.skipped = "silence"
        if res.skipped:
            res.warnings.append(f"No speech to transcribe ({res.skipped}); please repeat.")
            res.reliable = False
            return res

        order: List[str] = {"auto": ["cloud", "local"], "cloud": ["cloud"], "local": ["local"]}[mode]
        if mode == "auto" and hint in self.cfg.specialists and getattr(self.local, "available", False):
            order = ["local", "cloud"]          # a specialist model beats a stock cloud model for this language
        failures: List[str] = []
        er: Optional[EngineResult] = None
        for name in order:
            try:
                if name == "cloud":
                    ok, why = self._cloud_ready()
                    if not ok:
                        raise CloudError(why, skip=True)
                    er = self._run_cloud(x, hint, prompt, res)
                    self._cloud_ok()
                else:
                    er = self.local.transcribe(x, hint, prompt)
                break
            except (CloudError, LocalEngineError) as exc:
                if name == "cloud" and not getattr(exc, "skip", False):
                    self._cloud_failed()
                failures.append(f"{name}: {exc}")
                log.warning("STT engine '%s' failed: %s", name, exc)
        if er is None:
            raise STTError("all speech-to-text engines failed -> " + " | ".join(failures))

        # specialist routing: the cloud detected a language that has a better local model -> use that
        via_specialist = False
        if mode == "auto" and er.engine == "cloud" and hint is None and er.language in self.cfg.specialists \
                and getattr(self.local, "available", False):
            try:
                er = self._specialist_rerun(x, er, prompt)
                via_specialist = True
            except LocalEngineError as exc:
                res.warnings.append(f"Specialist model for '{er.language}' unavailable ({exc}); using the cloud result.")
                res.reliable = False
        # "fallback" = auto mode wanted its first-choice engine but another one produced the answer
        res.fallback_used = mode == "auto" and er.engine != order[0] and not via_specialist
        res.fallback_reason = failures[0] if res.fallback_used and failures else None
        res.engine, res.model = er.engine, er.model
        res.latency_s, res.segments = er.latency_s, er.segments
        res.language_probs, res.language_confidence = er.language_probs, er.language_confidence
        res.warnings.extend(er.notes)
        self._finalise(res, er, hint)
        return res

    def _run_cloud(self, x: np.ndarray, hint: Optional[str], prompt: Optional[str],
                   res: TranscriptionResult) -> EngineResult:
        er = self.cloud.transcribe(x, hint, prompt)
        if hint is None and er.language not in self.cfg.supported:
            guess = language_from_script(er.text) or _CONFUSION.get(er.language, "en")
            res.warnings.append(f"Cloud detected '{er.language}', which is not supported; "
                                f"retrying as '{guess}'.")
            er2 = self.cloud.transcribe(x, guess, prompt)
            er2.language = guess
            er2.latency_s += er.latency_s
            return er2
        return er

    def _specialist_rerun(self, x: np.ndarray, cloud_er: EngineResult, prompt: Optional[str]) -> EngineResult:
        er2 = self.local.transcribe(x, cloud_er.language, prompt)
        er2.notes.insert(0, f"Cloud detected '{cloud_er.language}'; transcribed with the local specialist model instead.")
        er2.latency_s += cloud_er.latency_s
        return er2

    def _finalise(self, res: TranscriptionResult, er: EngineResult, hint: Optional[str]) -> None:
        def doubt(msg: str) -> None:
            res.warnings.append(msg)
            res.reliable = False

        text = _clean_text(er.text)
        if text and _is_repetition_loop(text):
            doubt("The recogniser produced a repeating loop (likely noise); output discarded.")
            text = ""
        res.text = text
        script = language_from_script(text) if text else None
        lang = hint or er.language
        if hint:
            if script and script != hint and script != "en":
                doubt(f"Text looks like '{script}' but language '{hint}' was requested.")
        else:
            if lang not in self.cfg.supported:
                lang = script or _CONFUSION.get(lang, "en")
                res.warnings.append(f"Language outside the supported four; using '{lang}'.")
            elif script in ("ml", "kn", "hi") and script != lang:
                doubt(f"Language corrected from '{lang}' to '{script}' (writing system of the text).")
                lang = script
            elif script == "en" and lang in ("ml", "kn", "hi") and text:
                doubt(f"Detected '{lang}' but the text is in Latin letters (romanised or English).")
        if hint and text and script == "en" and hint in ("ml", "kn", "hi"):
            doubt(f"Asked for '{hint}' but the text is in Latin letters (possibly translated to English).")
        if not text:
            doubt("No speech recognised; please repeat.")
            if not hint and er.language not in self.cfg.supported:
                lang = UNKNOWN
        res.language = lang
        conf = res.language_confidence
        if conf is not None and conf < self.cfg.low_language_confidence and not hint:
            doubt(f"Low language confidence ({conf:.2f}); consider asking the user to choose the language.")
        if er.supported_mass is not None and er.supported_mass < 0.3 and not hint:
            doubt("The speech may be in a language other than Malayalam, Kannada, Hindi or English.")
        if (er.engine == "local" and lang in _WEAK_LANGUAGES and text
                and os.path.basename(er.model.split(":", 1)[-1].rstrip("/\\")) in _WEAK_LOCAL_MODELS):
            doubt(f"The local Whisper '{er.model.split(':', 1)[-1]}' model is unreliable for {SUPPORTED_LANGUAGES[lang]}; "
                  "use the cloud engine or a larger local model, and confirm the result with the user.")

    # ---- convenience ------------------------------------------------------------------
    def transcribe(self, source: Any, sample_rate: Optional[int] = None, **kw: Any) -> Tuple[str, str]:
        """The 2A contract: ``text, language = transcribe(audio)``."""
        return self.transcribe_full(source, sample_rate, **kw).as_tuple()

    def detect_language(self, source: Any, sample_rate: Optional[int] = None,
                        preprocess: bool = False) -> Tuple[str, float, Dict[str, float]]:
        """Language ID only.  Local Whisper's language probabilities (restricted to the four
        languages) when available, otherwise a cloud transcription.  Returns ``(code, confidence, probs)``."""
        x = self._prepare(source, sample_rate, preprocess)
        if hasattr(self.local, "detect_language") and getattr(self.local, "available", False):
            try:
                best, conf, probs, _ = self.local.detect_language(x)
                return best, conf, probs
            except LocalEngineError as exc:
                log.warning("local language ID failed: %s", exc)
        r = self.transcribe_full(x, engine="cloud")
        return r.language, r.language_confidence or 0.0, {r.language: 1.0}


# =========================================================================== module-level API
_default: Optional[SpeechToText] = None
_default_lock = threading.Lock()


def get_default() -> SpeechToText:
    global _default
    with _default_lock:
        if _default is None:
            _default = SpeechToText()
        return _default


def reset_default() -> None:
    global _default
    with _default_lock:
        _default = None


def transcribe(source: Any, sample_rate: Optional[int] = None, **kw: Any) -> Tuple[str, str]:
    """``text, language = transcribe(audio)``.  Keywords: ``language`` (hint: 'ml'/'kn'/'hi'/'en'),
    ``prompt``, ``engine`` ('auto'|'cloud'|'local'), ``preprocess`` (run 1B first)."""
    return get_default().transcribe(source, sample_rate, **kw)


def transcribe_full(source: Any, sample_rate: Optional[int] = None, **kw: Any) -> TranscriptionResult:
    """Like :func:`transcribe` but returns the full :class:`TranscriptionResult`."""
    return get_default().transcribe_full(source, sample_rate, **kw)


def detect_language(source: Any, sample_rate: Optional[int] = None, **kw: Any):
    return get_default().detect_language(source, sample_rate, **kw)


# =========================================================================== CLI
def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse

    p = argparse.ArgumentParser(description="SVARA 2A: speech-to-text with cloud + local Whisper.")
    p.add_argument("audio", nargs="?", help="audio file to transcribe")
    p.add_argument("--engine", choices=["auto", "cloud", "local"], default=None)
    p.add_argument("--language", choices=list(SUPPORTED_LANGUAGES), help="skip language detection")
    p.add_argument("--preprocess", action="store_true", help="run the 1B cleaning chain first")
    p.add_argument("--download", action="store_true", help="download/cache the local Whisper model, then exit")
    p.add_argument("--status", action="store_true", help="show which engines are available, then exit")
    a = p.parse_args(argv)
    stt = SpeechToText()
    if a.status:
        c, l = stt.cloud, stt.local
        print(f"cloud : {'READY' if c.available else 'not configured'} "
              f"(provider={c.provider}, model={getattr(c, 'model', None)}; set GROQ_API_KEY or OPENAI_API_KEY)")
        print(f"local : {'faster-whisper installed' if l.available else 'faster-whisper NOT installed'} "
              f"(model={stt.cfg.local_model})")
        return 0
    if a.download:
        try:
            stt.warm_up()
        except LocalEngineError as exc:
            print("error:", exc); return 1
        print(f"Whisper '{stt.cfg.local_model}' is ready for offline use.")
        return 0
    if not a.audio:
        p.error("give an audio file (or use --download / --status)")
    try:
        r = stt.transcribe_full(a.audio, engine=a.engine, language=a.language, preprocess=a.preprocess)
    except (AudioLoadError, STTError, ValueError) as exc:
        print("error:", exc); return 1
    print(f"engine   : {r.engine} ({r.model}){'  [FALLBACK: ' + str(r.fallback_reason) + ']' if r.fallback_used else ''}")
    print(f"language : {r.language}" + (f" (confidence {r.language_confidence:.2f})" if r.language_confidence is not None else ""))
    print(f"latency  : {r.latency_s:.2f} s for {r.audio_seconds:.1f} s of audio")
    print(f"text     : {r.text}")
    for w in r.warnings:
        print("WARNING  :", w)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
