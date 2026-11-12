"""
preprocess.py  --  SVARA, Member 1 / Sub-module 1B: Preprocessing Chain
=======================================================================

Public contract (consumed by Member 2's ``transcribe`` and by 1C's plots)::

    cleaned = clean_audio(source)          # float32, mono, 16 kHz, peak 0.9

``source`` is anything sub-module 1A (``audio_io.load_audio``) accepts -- a file path,
bytes, an upload, a ``(signal, sr)`` tuple -- or simply the 1-D array that 1A returned
(a bare array is assumed to be 16 kHz).

The chain, in order
-------------------
1. **Pre-emphasis**       y[n] = x[n] - 0.97 x[n-1]        (first-order high-pass tilt)
2. **Band-pass filter**   300-3400 Hz Butterworth, zero-phase (forward-backward)
3. **Noise reduction**    spectral subtraction (own implementation) or ``noisereduce``
4. **VAD trimming**       energy-based voice-activity detection removes leading and
                          trailing silence (with a safety margin)
5. **Re-normalise**       peak 0.9 again, so the ASR sees a consistent level

Every stage can be switched off / tuned through :class:`PreprocessConfig`; that is what
1C needs for "with vs without preprocessing" and for stage-by-stage ablations.
:func:`clean_audio_with_stages` additionally returns the signal after each stage, the VAD
decision and a report -- exactly what the Signal Lab tab needs for before/after plots.

Design notes (what is deliberately *not* done)
----------------------------------------------
* The filter is zero-phase, so no stage delays the speech; VAD times map directly onto the
  input.  Offline processing makes this possible (a live system would need a causal filter).
* Only leading/trailing silence is trimmed; pauses *inside* the utterance are kept, because
  cutting them changes how numbers and names are spoken ("nine ... five").
* If no reliable speech/noise separation is possible, the audio is returned untrimmed and
  the report says so, rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from audio_io import (TARGET_SR, AudioLoadError, AudioReport, load_audio_with_report,
                      normalize_amplitude)

log = logging.getLogger(__name__)

__all__ = [
    "PreprocessConfig", "PreprocessReport", "PreprocessResult", "VadResult",
    "clean_audio", "clean_audio_with_stages",
    "pre_emphasis", "bandpass_filter", "spectral_subtraction", "reduce_noise",
    "detect_speech", "trim_silence", "frame_energy_db",
    "pre_emphasis_response", "bandpass_response",
]

_EPS = 1e-12


# =========================================================================== config
@dataclass(frozen=True)
class PreprocessConfig:
    """All tunable settings of the chain.  Defaults follow the project specification."""

    # 1. pre-emphasis ----------------------------------------------------------------
    pre_emphasis: Optional[float] = 0.97       # coefficient alpha; None or 0 = off
    # 2. band-pass -------------------------------------------------------------------
    bandpass: bool = True
    low_hz: float = 300.0
    high_hz: float = 3400.0
    filter_order: int = 4                      # per edge; band-pass has 2*order poles
    # 3. noise reduction -------------------------------------------------------------
    denoise: Optional[str] = "spectral_subtraction"   # | "noisereduce" | None
    ss_n_fft: int = 512                        # 32 ms window at 16 kHz
    ss_hop: int = 128                          # 8 ms hop (75 % overlap)
    ss_alpha: float = 2.0                      # over-subtraction factor
    ss_gain_floor: float = 0.1                 # minimum gain (-20 dB): avoids "musical noise"
    ss_noise_percentile: float = 20.0          # quietest 20 % of frames = noise estimate
    ss_min_range_db: float = 3.0               # need this much loud/quiet contrast
    ss_smooth_frames: int = 3                  # temporal smoothing of the gain
    # 4. voice activity detection ----------------------------------------------------
    vad: bool = True
    vad_frame_ms: float = 25.0
    vad_hop_ms: float = 10.0
    vad_threshold_fraction: float = 0.3        # threshold = floor + 30 % of (speech - floor)
    vad_min_margin_db: float = 6.0             # ... but at least 6 dB above (or half the contrast)
    vad_min_dynamic_range_db: float = 3.0      # below this: "cannot tell speech from noise"
    vad_min_speech_ms: float = 80.0            # shorter bursts (clicks) are dropped
    vad_min_silence_ms: float = 300.0          # shorter gaps inside speech are bridged
    vad_pad_ms: float = 150.0                  # margin kept around the speech
    # 5. output level ----------------------------------------------------------------
    normalize_output: Optional[str] = "peak"   # "peak" | "rms" | None
    target_peak: float = 0.9

    def validate(self, sr: int) -> None:
        nyq = sr / 2.0
        if self.pre_emphasis not in (None, 0) and not 0.0 < float(self.pre_emphasis) < 1.0:
            raise ValueError("pre_emphasis must be in (0, 1), e.g. 0.97 (or None to disable)")
        if self.bandpass:
            if not 0 < self.low_hz < self.high_hz:
                raise ValueError("need 0 < low_hz < high_hz")
            if self.low_hz >= nyq:
                raise ValueError(f"low_hz must be below the Nyquist frequency {nyq:.0f} Hz")
            if not 1 <= int(self.filter_order) <= 10:
                raise ValueError("filter_order must be between 1 and 10")
        if self.denoise not in (None, "spectral_subtraction", "noisereduce"):
            raise ValueError("denoise must be 'spectral_subtraction', 'noisereduce' or None")
        if self.ss_n_fft < 64 or self.ss_hop < 1 or self.ss_hop > self.ss_n_fft // 2:
            raise ValueError("need ss_n_fft >= 64 and 1 <= ss_hop <= ss_n_fft/2")
        if not 0.0 <= self.ss_gain_floor < 1.0:
            raise ValueError("ss_gain_floor must be in [0, 1)")
        if self.ss_alpha < 0:
            raise ValueError("ss_alpha must be >= 0")
        if not 0.0 < self.vad_threshold_fraction < 1.0:
            raise ValueError("vad_threshold_fraction must be in (0, 1)")
        if min(self.vad_frame_ms, self.vad_hop_ms) <= 0 or self.vad_hop_ms > self.vad_frame_ms:
            raise ValueError("need 0 < vad_hop_ms <= vad_frame_ms")


# =========================================================================== stage 1
def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """First-order high-frequency boost: ``y[n] = x[n] - alpha * x[n-1]`` (y[0] = x[0]).

    Speech has a spectral tilt of about -6 to -12 dB/octave: most energy sits in the low
    harmonics, so weak high-frequency consonant cues (s, sh, f, t, k) are tiny next to the
    vowels.  The filter's gain rises with frequency (|H| = 0.03 at DC, 1.0 near 2.7 kHz,
    1.97 at 8 kHz), flattening the spectrum so later stages treat all bands alike and
    high-frequency information is not lost in rounding/noise.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - np.float32(alpha) * x[:-1]
    return y


def pre_emphasis_response(alpha: float = 0.97, sr: int = TARGET_SR, n: int = 512):
    """(frequencies in Hz, gain in dB) of the pre-emphasis filter -- for plotting."""
    w, h = sps.freqz([1.0, -alpha], [1.0], worN=n, fs=sr)
    return w, 20 * np.log10(np.abs(h) + _EPS)


# =========================================================================== stage 2
def _bandpass_sos(low: float, high: float, order: int, sr: int) -> np.ndarray:
    nyq = sr / 2.0
    high = min(high, nyq * 0.99)                 # keep the upper edge below Nyquist
    if high <= low:
        raise ValueError("band-pass upper edge must exceed the lower edge")
    return sps.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")


def bandpass_filter(x: np.ndarray, sr: int = TARGET_SR, low: float = 300.0,
                    high: float = 3400.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (``sosfiltfilt``).

    *Butterworth*: maximally flat pass-band (no ripple that would colour the voice).
    *Second-order sections*: numerically stable at this narrow band.
    *Forward-backward*: the signal is filtered, reversed, filtered again and reversed;
    phase distortion cancels (no time shift) and the attenuation doubles (-3 dB edges
    become -6 dB).  The 300-3400 Hz band is the classic telephone band that carries the
    vowel formants and most consonant information; it removes mains hum / rumble / handling
    noise below 300 Hz and hiss above 3.4 kHz.
    """
    x = np.asarray(x, dtype=np.float32)
    sos = _bandpass_sos(low, high, order, sr)
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.size <= padlen:                         # too short for forward-backward padding
        return np.asarray(sps.sosfilt(sos, x), dtype=np.float32)
    return np.asarray(sps.sosfiltfilt(sos, x), dtype=np.float32)


def bandpass_response(sr: int = TARGET_SR, low: float = 300.0, high: float = 3400.0,
                      order: int = 4, n: int = 2048):
    """(frequencies in Hz, gain in dB) of the *zero-phase* (forward-backward) band-pass."""
    sos = _bandpass_sos(low, high, order, sr)
    w, h = sps.sosfreqz(sos, worN=n, fs=sr)
    return w, 20 * np.log10(np.abs(h) ** 2 + _EPS)   # filtfilt squares the magnitude


# =========================================================================== stage 3
def frame_energy_db(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Short-time energy per frame in dB (10 log10 of mean square).

    Computed in O(n) with a cumulative sum of squares.  The tail is zero-padded so the
    last partial frame is included.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    n_frames = 1 + max(0, int(np.ceil((n - frame_len) / hop)))
    need = (n_frames - 1) * hop + frame_len
    xp = np.pad(x, (0, max(0, need - n)))
    cs = np.concatenate([[0.0], np.cumsum(xp * xp)])
    starts = np.arange(n_frames) * hop
    e = (cs[starts + frame_len] - cs[starts]) / frame_len
    return 10.0 * np.log10(np.maximum(e, 0.0) + _EPS)


def spectral_subtraction(x: np.ndarray, sr: int = TARGET_SR, *, n_fft: int = 512,
                         hop: int = 128, alpha: float = 2.0, gain_floor: float = 0.1,
                         noise_percentile: float = 20.0, min_range_db: float = 3.0,
                         smooth_frames: int = 3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Noise reduction by (over-)subtracting an estimated noise power spectrum.

    1. STFT (Hann window, 75 % overlap):  X(t, f).
    2. Noise estimate: average the power spectra of the quietest ``noise_percentile`` %
       of frames, i.e. the pauses before/after the words (no separate "silence" recording
       is needed).
    3. Per-bin gain (Boll's spectral subtraction with Berouti over-subtraction):
           G(t,f)^2 = max( 1 - alpha * N(f) / |X(t,f)|^2 ,  floor^2 )
       Bins dominated by noise are pulled down to the floor, speech-dominated bins stay
       almost unchanged.
    4. The gain is smoothed over ``smooth_frames`` frames (suppresses isolated random
       peaks that sound like "musical noise"), applied to the *complex* STFT (the noisy
       phase is reused) and the signal is rebuilt by inverse STFT / overlap-add.

    Returns ``(denoised, info)``.  If the clip has no pauses (loudest and quietest frames
    differ by less than ``min_range_db``, default 3 dB) a reliable noise estimate is impossible and the
    input is returned unchanged with ``info['applied'] = False``.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    info: Dict[str, Any] = {"method": "spectral_subtraction", "applied": False, "reason": ""}
    if n < n_fft:
        info["reason"] = "clip shorter than one analysis window"
        return x, info
    _, _, Z = sps.stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                       boundary="zeros", padded=True)
    P = np.abs(Z) ** 2
    e_db = 10 * np.log10(P.sum(axis=0) + _EPS)
    rng = float(np.percentile(e_db, 95) - np.percentile(e_db, 10))
    info["energy_range_db"] = rng
    if rng < min_range_db:
        info["reason"] = (f"no clear pauses (energy range {rng:.1f} dB < {min_range_db:.0f} dB)"
                          "; noise level cannot be estimated")
        return x, info
    cutoff = np.percentile(e_db, noise_percentile)
    idx = e_db <= cutoff
    if idx.sum() < 3:
        idx = np.argsort(e_db)[: min(3, e_db.size)]
    Pn = P[:, idx].mean(axis=1, keepdims=True)
    gain = np.sqrt(np.clip(1.0 - alpha * Pn / (P + _EPS), gain_floor ** 2, 1.0))
    if smooth_frames > 1:
        gain = uniform_filter1d(gain, size=smooth_frames, axis=1, mode="nearest")
        gain = np.maximum(gain, gain_floor)
    _, y = sps.istft(gain * Z, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                     boundary=True)
    y = np.asarray(y[:n], dtype=np.float32)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    info.update(applied=True, noise_frames=int(idx.sum()), total_frames=int(e_db.size),
                noise_power_db=float(10 * np.log10(Pn.sum() + _EPS)))
    return y, info


def _noisereduce(x: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"method": "noisereduce", "applied": False, "reason": ""}
    try:
        import noisereduce as nr
    except ImportError:
        raise ImportError(
            "denoise='noisereduce' needs the 'noisereduce' package (pip install noisereduce); "
            "use denoise='spectral_subtraction' for the built-in method"
        ) from None
    y = nr.reduce_noise(y=np.asarray(x, dtype=np.float32), sr=sr, stationary=True,
                        prop_decrease=1.0)
    info["applied"] = True
    return np.asarray(y, dtype=np.float32), info


def reduce_noise(x: np.ndarray, sr: int = TARGET_SR, method: Optional[str] = "spectral_subtraction",
                 cfg: Optional[PreprocessConfig] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dispatch to the chosen noise-reduction method.  Returns ``(signal, info)``."""
    cfg = cfg or PreprocessConfig()
    if method is None:
        return np.asarray(x, dtype=np.float32), {"method": None, "applied": False,
                                                 "reason": "disabled"}
    if method == "spectral_subtraction":
        return spectral_subtraction(
            x, sr, n_fft=cfg.ss_n_fft, hop=cfg.ss_hop, alpha=cfg.ss_alpha,
            gain_floor=cfg.ss_gain_floor, noise_percentile=cfg.ss_noise_percentile,
            min_range_db=cfg.ss_min_range_db, smooth_frames=cfg.ss_smooth_frames)
    if method == "noisereduce":
        return _noisereduce(x, sr)
    raise ValueError(f"unknown noise-reduction method {method!r}")


# =========================================================================== stage 4
@dataclass
class VadResult:
    """Outcome of voice-activity detection on one clip."""

    frame_times_s: np.ndarray          # centre time of each analysis frame
    energy_db: np.ndarray              # short-time energy per frame
    threshold_db: float                # decision threshold
    noise_floor_db: float              # 10th percentile of frame energy
    speech_level_db: float             # 95th percentile of frame energy
    mask: np.ndarray                   # bool per frame, after clean-up
    raw_mask: np.ndarray               # bool per frame, before clean-up
    segments: List[Tuple[int, int]]    # speech segments as (start, end) sample indices
    reliable: bool                     # False if speech/noise contrast was too small
    speech_detected: bool


def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    """Run-length encode a boolean array as (start, end_exclusive, value)."""
    if mask.size == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [mask.size]])
    return [(int(s), int(e), bool(mask[s])) for s, e in zip(starts, ends)]


def _clean_mask(raw: np.ndarray, min_speech: int, min_silence: int) -> np.ndarray:
    """Remove speech bursts shorter than ``min_speech`` frames, then bridge silences
    shorter than ``min_silence`` frames that lie *between* speech (hangover)."""
    m = raw.copy()
    for s, e, v in _runs(m):
        if v and (e - s) < min_speech:
            m[s:e] = False
    for s, e, v in _runs(m):
        if not v and (e - s) < min_silence and s > 0 and e < m.size:
            m[s:e] = True
    return m


def detect_speech(x: np.ndarray, sr: int = TARGET_SR,
                  cfg: Optional[PreprocessConfig] = None) -> VadResult:
    """Energy-based voice activity detection with an adaptive threshold.

    1. Cut the signal into 25 ms frames every 10 ms and compute each frame's energy in dB.
    2. Estimate the background level (10th percentile of the energies) and the speech level
       (95th percentile).  The threshold sits ``vad_threshold_fraction`` (30 %) of the way
       from background to speech, but never closer than 6 dB to the background (or half
       the contrast, if the contrast is below 12 dB).  It adapts
       to every recording, so no fixed loudness has to be assumed.
    3. frame is speech  <=>  energy > threshold.
    4. Clean-up ("hangover"): bursts shorter than 80 ms (clicks, pops) are discarded; gaps
       shorter than 300 ms between speech (stops, between words) are bridged.
    5. If the speech/background contrast is under 3 dB nothing can be decided reliably:
       ``reliable=False`` and the whole clip is treated as speech (so it is never cut).
    """
    cfg = cfg or PreprocessConfig()
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    frame_len = max(1, int(round(cfg.vad_frame_ms * sr / 1000)))
    hop = max(1, int(round(cfg.vad_hop_ms * sr / 1000)))
    e = frame_energy_db(x, frame_len, hop)
    times = (np.arange(e.size) * hop + frame_len / 2) / sr
    floor = float(np.percentile(e, 10))
    top = float(np.percentile(e, 95))
    dyn = top - floor
    if dyn < cfg.vad_min_dynamic_range_db:
        mask = np.ones(e.size, bool)
        return VadResult(times, e, floor, floor, top, mask, mask.copy(),
                         [(0, n)], False, False)
    margin = min(cfg.vad_min_margin_db, 0.5 * dyn)   # low-contrast clips: half the contrast
    thr = floor + max(margin, cfg.vad_threshold_fraction * dyn)
    raw = e > thr
    min_speech = max(1, int(round(cfg.vad_min_speech_ms / cfg.vad_hop_ms)))
    min_sil = max(1, int(round(cfg.vad_min_silence_ms / cfg.vad_hop_ms)))
    mask = _clean_mask(raw, min_speech, min_sil)
    segs = [(s * hop, min(n, (e_ - 1) * hop + frame_len)) for s, e_, v in _runs(mask) if v]
    return VadResult(times, e, float(thr), floor, top, mask, raw, segs, True, bool(segs))


def trim_silence(x: np.ndarray, vad: VadResult, sr: int = TARGET_SR,
                 pad_ms: float = 150.0) -> Tuple[np.ndarray, int, int]:
    """Cut leading/trailing non-speech, keeping ``pad_ms`` of margin.  Returns
    ``(trimmed, start_sample, end_sample)``.  Without detected speech: unchanged."""
    n = len(x)
    if not vad.speech_detected or not vad.segments:
        return x, 0, n
    pad = int(round(pad_ms * sr / 1000))
    start = max(0, vad.segments[0][0] - pad)
    end = min(n, vad.segments[-1][1] + pad)
    return x[start:end], start, end


# =========================================================================== pipeline
@dataclass
class PreprocessReport:
    stages_applied: List[str] = field(default_factory=list)
    input_duration_s: float = 0.0
    output_duration_s: float = 0.0
    trimmed_start_s: float = 0.0
    trimmed_end_s: float = 0.0
    noise_reduction_method: Optional[str] = None
    noise_reduction_applied: bool = False
    noise_floor_before_db: float = 0.0     # background level after band-pass
    noise_floor_after_db: float = 0.0      # background level after noise reduction
    speech_detected: bool = False
    vad_reliable: bool = False
    n_speech_segments: int = 0
    audio_input: Optional[AudioReport] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["audio_input"] = self.audio_input.to_dict() if self.audio_input else None
        return d


@dataclass
class PreprocessResult:
    """Everything 1C / the Signal Lab needs.  ``stages`` signals share the input's time axis
    (same length) except ``"trimmed"`` (the final output before re-normalisation)."""

    signal: np.ndarray                         # final cleaned audio
    sr: int
    stages: Dict[str, np.ndarray]              # input, pre_emphasis, bandpass, denoised, trimmed
    vad: Optional[VadResult]
    trim_start: int
    trim_end: int
    report: PreprocessReport
    config: PreprocessConfig


def _frames_floor_db(x: np.ndarray, sr: int, cfg: PreprocessConfig) -> float:
    fl = max(1, int(round(cfg.vad_frame_ms * sr / 1000)))
    hp = max(1, int(round(cfg.vad_hop_ms * sr / 1000)))
    return float(np.percentile(frame_energy_db(x, fl, hp), 10))


def clean_audio_with_stages(source: Any, sample_rate: Optional[int] = None,
                            config: Optional[PreprocessConfig] = None,
                            **overrides: Any) -> PreprocessResult:
    """Run the full chain and keep every intermediate result.  See module docstring.

    ``overrides`` are :class:`PreprocessConfig` fields, e.g.
    ``clean_audio_with_stages(f, denoise=None, vad=False)``.
    """
    cfg = replace(config or PreprocessConfig(), **overrides)
    if isinstance(source, np.ndarray) and sample_rate is None:
        sample_rate = TARGET_SR                       # a bare array is 1A output
    x, sr, in_rep = load_audio_with_report(source, sample_rate)   # 1A: mono, 16 kHz, 0.9 peak
    cfg.validate(sr)

    rep = PreprocessReport(input_duration_s=len(x) / sr, audio_input=in_rep,
                           warnings=list(in_rep.warnings))
    stages: Dict[str, np.ndarray] = {"input": x.copy()}
    y = x.copy()

    # 1) pre-emphasis
    if cfg.pre_emphasis:
        y = pre_emphasis(y, float(cfg.pre_emphasis))
        rep.stages_applied.append(f"pre_emphasis({cfg.pre_emphasis})")
    stages["pre_emphasis"] = y.copy()

    # 2) band-pass
    if cfg.bandpass:
        y = bandpass_filter(y, sr, cfg.low_hz, cfg.high_hz, cfg.filter_order)
        rep.stages_applied.append(f"bandpass({cfg.low_hz:.0f}-{cfg.high_hz:.0f} Hz)")
    stages["bandpass"] = y.copy()
    rep.noise_floor_before_db = _frames_floor_db(y, sr, cfg)

    # 3) noise reduction
    rep.noise_reduction_method = cfg.denoise
    if cfg.denoise:
        y, info = reduce_noise(y, sr, cfg.denoise, cfg)
        rep.noise_reduction_applied = bool(info.get("applied"))
        if rep.noise_reduction_applied:
            rep.stages_applied.append(f"noise_reduction({cfg.denoise})")
        elif info.get("reason"):
            rep.warnings.append(f"Noise reduction skipped: {info['reason']}.")
    rep.noise_floor_after_db = _frames_floor_db(y, sr, cfg)
    stages["denoised"] = y.copy()

    # 4) VAD trimming
    vad: Optional[VadResult] = None
    start, end = 0, len(y)
    if cfg.vad:
        vad = detect_speech(y, sr, cfg)
        rep.vad_reliable = vad.reliable
        rep.speech_detected = vad.speech_detected
        rep.n_speech_segments = len(vad.segments) if vad.speech_detected else 0
        if not vad.reliable:
            rep.warnings.append("Could not separate speech from background noise "
                                "(too little contrast); the clip was left untrimmed.")
        elif not vad.speech_detected:
            rep.warnings.append("No speech detected; please repeat.")
        y, start, end = trim_silence(y, vad, sr, cfg.vad_pad_ms)
        if vad.speech_detected and vad.reliable:
            rep.stages_applied.append(f"vad_trim(pad {cfg.vad_pad_ms:.0f} ms)")
    stages["trimmed"] = y.copy()
    rep.trimmed_start_s = start / sr
    rep.trimmed_end_s = (len(stages["denoised"]) - end) / sr

    # 5) output level
    if cfg.normalize_output:
        y, _ = normalize_amplitude(y, cfg.normalize_output, cfg.target_peak)
        rep.stages_applied.append(f"normalize({cfg.normalize_output})")
    y = np.ascontiguousarray(np.clip(y, -1.0, 1.0), dtype=np.float32)
    rep.output_duration_s = len(y) / sr
    for w in rep.warnings:
        log.warning(w)
    return PreprocessResult(y, sr, stages, vad, start, end, rep, cfg)


def clean_audio(source: Any, sample_rate: Optional[int] = None,
                config: Optional[PreprocessConfig] = None, **overrides: Any) -> np.ndarray:
    """The 1B contract: ``cleaned = clean_audio(source)`` (float32, mono, 16 kHz).

    ``source``: anything :func:`audio_io.load_audio` accepts, or a 1-D 16 kHz array.
    Keyword ``overrides`` are :class:`PreprocessConfig` fields, e.g.
    ``clean_audio(f, denoise="noisereduce", vad=False)``.
    """
    return clean_audio_with_stages(source, sample_rate, config, **overrides).signal


# =========================================================================== CLI
def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse

    from audio_io import save_wav

    p = argparse.ArgumentParser(description="SVARA 1B: clean speech audio for recognition.")
    p.add_argument("input")
    p.add_argument("output", nargs="?", help="optional output .wav path")
    p.add_argument("--denoise", choices=["spectral_subtraction", "noisereduce", "none"],
                   default="spectral_subtraction")
    p.add_argument("--no-vad", action="store_true")
    p.add_argument("--no-bandpass", action="store_true")
    p.add_argument("--no-preemph", action="store_true")
    a = p.parse_args(argv)
    try:
        res = clean_audio_with_stages(
            a.input, denoise=None if a.denoise == "none" else a.denoise, vad=not a.no_vad,
            bandpass=not a.no_bandpass, pre_emphasis=None if a.no_preemph else 0.97)
    except (AudioLoadError, ValueError, ImportError) as exc:
        print(f"error: {exc}")
        return 1
    r = res.report
    print("stages      :", " -> ".join(r.stages_applied))
    print(f"duration    : {r.input_duration_s:.2f} s -> {r.output_duration_s:.2f} s "
          f"(trimmed {r.trimmed_start_s:.2f} s start, {r.trimmed_end_s:.2f} s end)")
    print(f"noise floor : {r.noise_floor_before_db:.1f} dB -> {r.noise_floor_after_db:.1f} dB")
    print(f"speech found: {r.speech_detected} ({r.n_speech_segments} segment(s))")
    for w in r.warnings:
        print("WARNING     :", w)
    if a.output:
        print("saved:", save_wav(res.signal, a.output, res.sr))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
