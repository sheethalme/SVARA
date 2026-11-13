"""
features.py  --  SVARA, Member 1 / Sub-module 1C: Feature Extraction
====================================================================

Public contract (consumed by 1C's plots, the Signal Lab tab and the analytics)::

    feats = extract_features(audio)        # dict of NumPy arrays on one shared frame grid

``audio`` is the 1-D array that 1A (``load_audio``) or 1B (``clean_audio``) returned
(assumed 16 kHz unless ``sr=`` is given), or anything 1A accepts (path, bytes, upload,
``(signal, sr)`` tuple).

What is computed
----------------
Every feature is computed on the **same frames**: 25 ms long, a new one every 10 ms
(frame *i* is centred at ``i*hop + frame_len/2``), so all arrays have ``n_frames`` columns
and share ``feats["times"]``.

=================  ======================  ==============================================
key                shape                   meaning
=================  ======================  ==============================================
``energy``         (T,)                    short-time energy, mean of x[n]^2 in the frame
``energy_db``      (T,)                    the same in dB (identical to 1B's VAD energy)
``zcr``            (T,)                    zero-crossing rate, sign changes per sample
``spectral_centroid`` (T,)                 "centre of mass" of the spectrum, Hz
``f0``             (T,)                    pitch in Hz (YIN), ``NaN`` where unvoiced
``voiced``         (T,) bool               voicing decision of the pitch tracker
``voiced_prob``    (T,)                    1 - YIN aperiodicity, in [0, 1]
``mel_spectrogram`` (n_mels, T)            power in 40 triangular mel bands
``mel_spectrogram_db`` (n_mels, T)         the same in dB
``mfcc``           (13, T)                 mel-frequency cepstral coefficients
``mfcc_delta``     (13, T)                 their first time-derivative (delta)
``mfcc_delta2``    (13, T)                 second derivative (delta-delta)
=================  ======================  ==============================================

plus ``times`` (s), ``mel_frequencies`` (band centres, Hz), ``sr``, ``frame_len``,
``hop`` and ``n_fft``.

Framing and windowing (the viva question)
-----------------------------------------
Speech changes continuously, but over 20-30 ms the vocal tract hardly moves, so a short
piece of it can be treated as *stationary* and analysed with a Fourier transform.  The
signal is therefore cut into overlapping frames (25 ms, hop 10 ms: 60 % overlap so no
event falls between frames).  Cutting a frame out with a "rectangular" window creates
jumps at its edges, which smear energy over all frequencies (*spectral leakage*).  Each
frame is therefore multiplied by a **Hamming window**
``w[n] = 0.54 - 0.46 cos(2 pi n / (N-1))`` that tapers it towards the edges before the FFT.
Time-domain features (energy, ZCR) use the plain frame: they measure the waveform itself.

MFCC computation (the other viva question)
------------------------------------------
1. (pre-emphasis -- already done by 1B; ``pre_emphasis=`` adds it for raw audio)
2. framing + Hamming window (above)
3. FFT -> power spectrum ``|X(k)|^2 / N``
4. 26 triangular filters spaced evenly on the **mel scale** ``mel = 2595 log10(1 + f/700)``
   (the ear resolves low frequencies finely and high ones coarsely) -> 26 band energies
5. logarithm (loudness is perceived logarithmically; turns a gain into an additive offset)
6. DCT-II -> decorrelates the 26 log energies; keep the first 13 coefficients
   (the smooth spectral envelope = vocal-tract shape; the higher ones = pitch ripple)
7. delta and delta-delta: slopes of each coefficient over +-2 frames

Pitch (F0) uses the YIN algorithm (de Cheveigne & Kawahara, 2002) on a longer 40 ms
window so that at least two periods of the lowest pitch (60 Hz) fit in it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np
from scipy.fft import dct

from audio_io import TARGET_SR, load_audio

log = logging.getLogger(__name__)

__all__ = [
    "FeatureConfig", "extract_features", "summarize_features",
    "frame_signal", "hamming_window", "short_time_energy", "zero_crossing_rate",
    "power_spectrogram", "spectral_centroid", "hz_to_mel", "mel_to_hz",
    "mel_filterbank", "mel_spectrogram", "mfcc", "delta", "pitch_yin",
]

_EPS = 1e-12
SILENCE_DB = -80.0            # mean-square level treated as digital silence (pitch gate)


# =========================================================================== config
@dataclass(frozen=True)
class FeatureConfig:
    """All tunable settings.  Defaults follow the project specification."""

    frame_ms: float = 25.0             # analysis frame length
    hop_ms: float = 10.0               # frame step
    n_fft: int = 512                   # FFT size (>= frame length; 512 = 32 ms at 16 kHz)
    pre_emphasis: Optional[float] = None   # 1B already applied 0.97; set it for raw audio
    # mel spectrogram (for display) ---------------------------------------------------
    n_mels: int = 40
    fmin: float = 0.0
    fmax: Optional[float] = None       # None = Nyquist
    # MFCC ----------------------------------------------------------------------------
    n_mfcc: int = 13
    mfcc_filters: int = 26             # filterbank used for MFCCs (classic HTK value)
    delta_width: int = 2               # delta regression over +-2 frames
    # pitch ---------------------------------------------------------------------------
    f0_min: float = 60.0               # Hz; covers low male voices
    f0_max: float = 500.0              # Hz; covers children
    pitch_frame_ms: float = 40.0       # >= 2 periods of f0_min
    yin_threshold: float = 0.15        # aperiodicity below this = voiced
    voicing_energy_db: float = 40.0    # frames > 40 dB below the loudest are unvoiced

    def validate(self, sr: int) -> None:
        if min(self.frame_ms, self.hop_ms) <= 0 or self.hop_ms > self.frame_ms:
            raise ValueError("need 0 < hop_ms <= frame_ms")
        if self.n_fft < int(round(self.frame_ms * sr / 1000)):
            raise ValueError("n_fft must be at least the frame length in samples")
        if self.pre_emphasis not in (None, 0) and not 0.0 < float(self.pre_emphasis) < 1.0:
            raise ValueError("pre_emphasis must be in (0, 1) or None")
        fmax = self.fmax or sr / 2
        if not 0 <= self.fmin < fmax <= sr / 2:
            raise ValueError("need 0 <= fmin < fmax <= sr/2")
        if self.n_mels < 1 or self.mfcc_filters < 2 or not 1 <= self.n_mfcc <= self.mfcc_filters:
            raise ValueError("need n_mels >= 1 and 1 <= n_mfcc <= mfcc_filters")
        if self.delta_width < 1:
            raise ValueError("delta_width must be >= 1")
        if not 0 < self.f0_min < self.f0_max < sr / 2:
            raise ValueError("need 0 < f0_min < f0_max < sr/2")
        if self.pitch_frame_ms * sr / 1000 < 2 * sr / self.f0_max:
            raise ValueError("pitch_frame_ms too short for f0_max")
        if not 0 < self.yin_threshold < 1:
            raise ValueError("yin_threshold must be in (0, 1)")


# =========================================================================== framing
def _n_frames(n: int, frame_len: int, hop: int) -> int:
    """Same frame count as 1B's ``frame_energy_db``: tail zero-padded, never 0 frames."""
    return 1 + max(0, int(np.ceil((n - frame_len) / hop)))


def frame_signal(x: np.ndarray, frame_len: int, hop: int,
                 n_frames: Optional[int] = None, centre_len: Optional[int] = None) -> np.ndarray:
    """Cut ``x`` into overlapping frames, shape ``(n_frames, frame_len)``.

    Frame *i* is centred at sample ``i*hop + centre_len/2`` (``centre_len`` defaults to
    ``frame_len``, i.e. frame *i* starts at ``i*hop``).  A larger ``frame_len`` with the
    standard ``centre_len`` gives longer frames on the *same* centres (used for pitch).
    Samples outside the signal are zeros.  The result is a read-only view.
    """
    x = np.asarray(x, dtype=np.float64)
    centre_len = frame_len if centre_len is None else centre_len
    if n_frames is None:
        n_frames = _n_frames(x.size, centre_len, hop)
    left = max(0, (frame_len - centre_len) // 2)
    need = (n_frames - 1) * hop + frame_len
    xp = np.pad(x, (left, max(0, need - left - x.size)))
    starts = np.arange(n_frames) * hop
    return np.lib.stride_tricks.sliding_window_view(xp, frame_len)[starts]


def hamming_window(n: int) -> np.ndarray:
    """``w[k] = 0.54 - 0.46 cos(2 pi k / (n-1))`` -- tapers a frame to reduce leakage."""
    if n == 1:
        return np.ones(1)
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


# =========================================================================== time domain
def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Mean square of each frame: ``E = (1/N) sum x[n]^2``.  Loud speech >> pauses."""
    return np.mean(np.square(frames), axis=1)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of neighbouring samples whose sign differs (0 = no crossings, 1 = every
    sample).  Voiced speech (vowels) crosses rarely; fricatives (s, sh) and hiss often.
    Exact zeros count as positive, so digital silence has ZCR 0."""
    s = frames >= 0
    return np.mean(s[:, 1:] != s[:, :-1], axis=1)


# =========================================================================== frequency domain
def power_spectrogram(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Hamming-windowed power spectrum ``|FFT|^2 / n_fft``, shape ``(n_fft//2+1, T)``."""
    w = hamming_window(frames.shape[1])
    spec = np.fft.rfft(frames * w, n=n_fft, axis=1)
    return (np.abs(spec) ** 2 / n_fft).T


def spectral_centroid(power: np.ndarray, sr: int) -> np.ndarray:
    """``sum f|X(f)| / sum |X(f)|`` per frame, in Hz (magnitude-weighted, 0 for silence).
    High for bright/hissy sounds, low for dark/voiced ones."""
    mag = np.sqrt(power)
    f = np.linspace(0, sr / 2, power.shape[0])
    total = mag.sum(axis=0)
    return np.where(total > _EPS, (f[:, None] * mag).sum(axis=0) / np.maximum(total, _EPS), 0.0)


def hz_to_mel(f):
    """HTK mel scale: ``2595 log10(1 + f/700)``.  ~linear below 1 kHz, ~log above."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_filters: int, fmin: float = 0.0,
                   fmax: Optional[float] = None):
    """Triangular filters evenly spaced on the mel scale.  Returns
    ``(weights[n_filters, n_fft//2+1], centre_frequencies_hz)``.

    Filter *m* rises from 0 at edge ``m`` to 1 at edge ``m+1`` and falls back to 0 at edge
    ``m+2``, where the ``n_filters+2`` edges are equally spaced in mel.  Peak height is 1
    (HTK convention), so a flat spectrum gives wider -- higher -- bands more energy.
    """
    fmax = sr / 2 if fmax is None else fmax
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2))
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    lo, ce, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (f - lo) / np.maximum(ce - lo, _EPS)
    down = (hi - f) / np.maximum(hi - ce, _EPS)
    return np.maximum(0.0, np.minimum(up, down)), edges[1:-1]


def mel_spectrogram(power: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Apply the filterbank: energy in each mel band per frame, shape ``(n_filters, T)``."""
    return fb @ power


def mfcc(power: np.ndarray, sr: int, n_fft: int, n_mfcc: int = 13, n_filters: int = 26,
         fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    """MFCCs from a power spectrogram: mel filterbank -> natural log -> DCT-II (orthonormal)
    -> first ``n_mfcc`` coefficients.  ``c0`` is the overall log energy of the frame."""
    fb, _ = mel_filterbank(sr, n_fft, n_filters, fmin, fmax)
    log_mel = np.log(mel_spectrogram(power, fb) + 1e-10)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]


def delta(c: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression slope over +-``width`` frames (HTK/ETSI formula), edges repeated:
    ``d_t = sum_n n (c_{t+n} - c_{t-n}) / (2 sum_n n^2)``."""
    T = c.shape[-1]
    padded = np.pad(c, [(0, 0)] * (c.ndim - 1) + [(width, width)], mode="edge")
    num = sum(n * (padded[..., width + n:width + n + T] - padded[..., width - n:width - n + T])
              for n in range(1, width + 1))
    return num / (2 * sum(n * n for n in range(1, width + 1)))


# =========================================================================== pitch
def pitch_yin(x: np.ndarray, sr: int = TARGET_SR, hop: int = 160, frame_len: int = 400,
              n_frames: Optional[int] = None, *, f0_min: float = 60.0, f0_max: float = 500.0,
              pitch_frame_ms: float = 40.0, threshold: float = 0.15,
              voicing_energy_db: float = 40.0):
    """YIN fundamental-frequency estimator.  Returns ``(f0, voiced, voiced_prob)``.

    For each frame (``pitch_frame_ms`` long, centred like the 25 ms feature frames):

    1. difference function ``d(tau) = sum_j (x_j - x_{j+tau})^2`` -- small when the signal
       repeats itself after ``tau`` samples.  Computed for all frames at once with FFTs
       (``d = E_0 + E_tau - 2 r(tau)``).
    2. cumulative mean normalisation ``d'(tau) = d(tau) * tau / sum_{k<=tau} d(k)`` --
       removes the trivial dip at tau = 0 and makes the values comparable (aperiodicity).
    3. absolute threshold: the first tau in [sr/f0_max, sr/f0_min] where ``d'`` drops below
       ``threshold``, followed down to its local minimum (avoids octave errors); if none,
       the global minimum is taken and the frame is called unvoiced.
    4. parabolic interpolation around the chosen tau for sub-sample accuracy;
       ``f0 = sr / tau``.

    Frames more than ``voicing_energy_db`` below the loudest frame are unvoiced as well
    (pauses would otherwise get a random pitch from the background noise).
    Unvoiced frames have ``f0 = NaN``.
    """
    x = np.asarray(x, dtype=np.float64)
    W = int(round(pitch_frame_ms * sr / 1000)) // 2         # integration window
    tau_min = max(2, int(np.floor(sr / f0_max)))
    tau_max = min(W, int(np.ceil(sr / f0_min)))
    L = W + tau_max                                        # samples needed per frame
    fr = frame_signal(x, L, hop, n_frames, centre_len=frame_len)
    T = fr.shape[0]

    # 1. difference function via FFT cross-correlation (all frames at once)
    nfft = int(2 ** np.ceil(np.log2(L + W)))
    A = np.fft.rfft(fr[:, :W], nfft, axis=1)
    B = np.fft.rfft(fr, nfft, axis=1)
    r = np.fft.irfft(np.conj(A) * B, nfft, axis=1)[:, :tau_max + 1]   # r(tau) = sum x_j x_{j+tau}
    cs = np.concatenate([np.zeros((T, 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e0 = cs[:, W:W + 1]                                    # energy of the first W samples
    et = cs[:, taus + W] - cs[:, taus]                     # energy of the window shifted by tau
    d = np.maximum(e0 + et - 2 * r, 0.0)

    # 2. cumulative mean normalised difference
    cmnd = np.ones_like(d)
    csum = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = np.where(csum > _EPS, d[:, 1:] * taus[1:] / np.maximum(csum, _EPS), 1.0)

    # 3. absolute threshold, then descend to the local minimum
    f0 = np.full(T, np.nan)
    aper = np.ones(T)
    voiced = np.zeros(T, bool)
    seg = cmnd[:, tau_min:tau_max + 1]
    for i in range(T):
        below = np.flatnonzero(seg[i] < threshold)
        if below.size:
            k = below[0]
            while k + 1 < seg.shape[1] and seg[i, k + 1] < seg[i, k]:
                k += 1
            voiced[i] = True
        else:
            k = int(np.argmin(seg[i]))
        tau = k + tau_min
        # 4. parabolic interpolation on d' around tau
        if tau_min < tau < tau_max:
            a, b, c = cmnd[i, tau - 1], cmnd[i, tau], cmnd[i, tau + 1]
            den = a - 2 * b + c
            shift = 0.5 * (a - c) / den if abs(den) > _EPS else 0.0
            tau_f = tau + float(np.clip(shift, -1, 1))
        else:
            tau_f = float(tau)
        aper[i] = float(np.clip(seg[i, k], 0.0, 1.0))
        f0[i] = sr / tau_f

    # silence gate: the frame's own energy relative to the loudest frame, and absolute
    # digital silence (below -80 dBFS, the same limit 1A uses for a silent clip)
    e_db = 10 * np.log10(e0[:, 0] / W + _EPS)
    voiced &= (e_db > e_db.max() - voicing_energy_db) & (e_db > SILENCE_DB)
    voiced &= (f0 >= f0_min * 0.95) & (f0 <= f0_max * 1.05)
    f0[~voiced] = np.nan
    return f0, voiced, 1.0 - aper


# =========================================================================== public API
def _as_signal(audio: Any, sr: Optional[int]):
    if isinstance(audio, np.ndarray):
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("extract_features expects a 1-D (mono) signal; use load_audio first")
        if x.size == 0:
            raise ValueError("cannot extract features from an empty signal")
        if not np.all(np.isfinite(x)):
            raise ValueError("signal contains NaN or infinite values")
        return x, int(sr or TARGET_SR)
    return load_audio(audio, sr)                            # path, bytes, upload, (x, sr)


def extract_features(audio: Any, sr: Optional[int] = None,
                     config: Optional[FeatureConfig] = None, **overrides: Any) -> Dict[str, Any]:
    """The 1C contract: ``feats = extract_features(audio)``.  See the module docstring.

    ``overrides`` are :class:`FeatureConfig` fields, e.g. ``extract_features(x, n_mfcc=20)``.
    An array is analysed as it is (no re-normalisation), so "before" and "after" signals
    can be compared directly.
    """
    cfg = replace(config or FeatureConfig(), **overrides)
    x, sr = _as_signal(audio, sr)
    cfg.validate(sr)
    if cfg.pre_emphasis:
        from preprocess import pre_emphasis
        x = pre_emphasis(x, float(cfg.pre_emphasis))

    frame_len = int(round(cfg.frame_ms * sr / 1000))
    hop = int(round(cfg.hop_ms * sr / 1000))
    T = _n_frames(x.size, frame_len, hop)
    frames = frame_signal(x, frame_len, hop, T)

    energy = short_time_energy(frames)
    power = power_spectrogram(frames, cfg.n_fft)
    fb, mel_f = mel_filterbank(sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    mel = mel_spectrogram(power, fb)
    c = mfcc(power, sr, cfg.n_fft, cfg.n_mfcc, cfg.mfcc_filters, cfg.fmin, cfg.fmax)
    d1 = delta(c, cfg.delta_width)
    f0, voiced, vprob = pitch_yin(
        x, sr, hop, frame_len, T, f0_min=cfg.f0_min, f0_max=cfg.f0_max,
        pitch_frame_ms=cfg.pitch_frame_ms, threshold=cfg.yin_threshold,
        voicing_energy_db=cfg.voicing_energy_db)

    return {
        "times": (np.arange(T) * hop + frame_len / 2) / sr,
        "energy": energy,
        "energy_db": 10 * np.log10(energy + _EPS),
        "zcr": zero_crossing_rate(frames),
        "spectral_centroid": spectral_centroid(power, sr),
        "f0": f0,
        "voiced": voiced,
        "voiced_prob": vprob,
        "mel_spectrogram": mel,
        "mel_spectrogram_db": 10 * np.log10(mel + 1e-10),
        "mel_frequencies": mel_f,
        "mfcc": c,
        "mfcc_delta": d1,
        "mfcc_delta2": delta(d1, cfg.delta_width),
        "sr": sr, "frame_len": frame_len, "hop": hop, "n_fft": cfg.n_fft,
    }


def summarize_features(feats: Dict[str, Any]) -> Dict[str, float]:
    """One number per feature -- for tables in the Signal Lab / analytics."""
    f0 = feats["f0"][feats["voiced"]]
    return {
        "duration_s": float(feats["times"][-1] + feats["frame_len"] / 2 / feats["sr"]),
        "n_frames": int(feats["times"].size),
        "mean_energy_db": float(np.mean(feats["energy_db"])),
        "energy_range_db": float(np.percentile(feats["energy_db"], 95)
                                 - np.percentile(feats["energy_db"], 10)),
        "mean_zcr": float(np.mean(feats["zcr"])),
        "mean_spectral_centroid_hz": float(np.mean(feats["spectral_centroid"])),
        "voiced_fraction": float(np.mean(feats["voiced"])),
        "median_f0_hz": float(np.median(f0)) if f0.size else float("nan"),
        "f0_range_hz": float(np.ptp(f0)) if f0.size else float("nan"),
    }
