"""
audio_io.py  --  SVARA, Member 1 / Sub-module 1A: Audio Input and Conditioning
=================================================================================

Public contract (consumed by 1B ``preprocess.clean_audio`` and by Member 3's app)::

    signal, sr = load_audio(source)        # float32, mono, 16 000 Hz, peak-normalised

What this module does, in order
-------------------------------
1. **Decode**   file path / bytes / file-like object / NumPy array / live microphone.
2. **Mono**     average all channels into one.
3. **DC removal**  subtract the mean (a constant offset carries no speech information).
4. **Resample** to 16 kHz with a band-limited (anti-aliasing) resampler.
5. **Normalise** the amplitude (peak by default) without ever amplifying digital silence.
6. **Validate** (finite values, sane duration) and build an :class:`AudioReport`.

Supported inputs
----------------
* ``str`` / ``os.PathLike``      -- WAV, FLAC, OGG, MP3 (libsndfile) and, via ffmpeg,
                                    WebM/Opus, M4A/AAC, MP4, AMR, ... (anything ffmpeg reads).
* ``bytes`` / ``bytearray`` / ``memoryview``  -- encoded audio held in memory.
* file-like object with ``.read()`` or ``.getvalue()``  -- e.g. Streamlit's
  ``st.audio_input`` / ``st.file_uploader`` objects, ``io.BytesIO``.
* ``numpy.ndarray`` plus ``sample_rate=`` -- or a ``(array, sr)`` / ``(sr, array)`` tuple
  (``scipy.io.wavfile.read`` returns the latter order).  Arrays are
  ``(samples,)``, ``(samples, channels)`` or ``(channels, samples)``; integer dtypes are
  scaled to [-1, 1].
* Microphone via :func:`record_from_mic` (needs the optional ``sounddevice`` package).

Every failure raises :class:`AudioLoadError` (a ``ValueError``) with a readable message so
the UI can show it to the user instead of crashing.
"""

from __future__ import annotations

import io
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

try:  # High-quality resampler (libsoxr).  Optional: scipy is the fallback.
    import soxr
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    soxr = None

log = logging.getLogger(__name__)

__all__ = [
    "TARGET_SR",
    "AudioLoadError",
    "AudioReport",
    "load_audio",
    "load_audio_with_report",
    "record_from_mic",
    "record_from_mic_with_report",
    "normalize_amplitude",
    "resample_audio",
    "save_wav",
]

# --------------------------------------------------------------------------- constants
TARGET_SR = 16_000            # Hz. What Whisper and most speech models expect.
DEFAULT_TARGET_PEAK = 0.9     # -0.9 dBFS: loud, but leaves head-room after resampling.
DEFAULT_TARGET_RMS = 0.1      # -20 dBFS: a typical speech loudness for RMS mode.
RMS_MODE_PEAK_LIMIT = 0.99    # RMS mode lowers its gain rather than hard-clipping.
SILENCE_PEAK = 1e-4           # Below this peak (-80 dBFS) the clip is treated as silence.
CLIP_LEVEL = 0.999            # |x| >= this counts as a clipped sample.
CLIP_WARN_FRACTION = 1e-3     # Warn if more than 0.1 % of samples are clipped.
HIGH_GAIN_WARN_DB = 40.0      # Warn if normalisation boosts the signal by more than this.
MIN_INPUT_SR, MAX_INPUT_SR = 1_000, 768_000
MAX_CHANNELS = 32
FFMPEG_TIMEOUT_S = 60


class AudioLoadError(ValueError):
    """Raised when audio cannot be decoded, is invalid, or violates the limits."""


# --------------------------------------------------------------------------- report
@dataclass
class AudioReport:
    """Facts about what happened to one clip.  Useful for the Signal Lab tab and logs."""

    source_type: str = ""            # path | bytes | file-like | array | microphone
    decoder: str = ""                # soundfile | ffmpeg | array
    original_sr: int = 0
    original_channels: int = 0
    original_duration_s: float = 0.0
    output_sr: int = TARGET_SR
    output_duration_s: float = 0.0
    resampled: bool = False
    dc_offset_removed: float = 0.0   # the mean that was subtracted (full-scale units)
    peak_before: float = 0.0         # after mono + DC removal, before normalisation
    rms_before: float = 0.0
    peak_after: float = 0.0
    rms_after: float = 0.0
    normalization: str = "none"
    gain_applied: float = 1.0        # linear
    gain_db: float = 0.0
    clipped_fraction_in: float = 0.0
    is_silent: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"source      : {self.source_type} (decoded by {self.decoder})",
            f"input       : {self.original_sr} Hz, {self.original_channels} ch, "
            f"{self.original_duration_s:.3f} s",
            f"output      : {self.output_sr} Hz, mono, {self.output_duration_s:.3f} s"
            f"{' (resampled)' if self.resampled else ''}",
            f"DC removed  : {self.dc_offset_removed:+.6f}",
            f"level before: peak {self.peak_before:.4f}, rms {self.rms_before:.4f}",
            f"level after : peak {self.peak_after:.4f}, rms {self.rms_after:.4f} "
            f"(mode={self.normalization}, gain {self.gain_db:+.1f} dB)",
            f"clipped in  : {self.clipped_fraction_in * 100:.3f} % of samples",
            f"silent      : {self.is_silent}",
        ]
        lines += [f"WARNING     : {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- decoding
def _check_sample_rate(sr: Any, what: str = "sample rate") -> int:
    try:
        sr_f = float(sr)
    except (TypeError, ValueError):
        raise AudioLoadError(f"Invalid {what}: {sr!r}") from None
    if not math.isfinite(sr_f) or sr_f != int(sr_f):
        raise AudioLoadError(f"Invalid {what}: {sr!r} (must be a whole number of Hz)")
    sr_i = int(sr_f)
    if not MIN_INPUT_SR <= sr_i <= MAX_INPUT_SR:
        raise AudioLoadError(
            f"Unsupported {what} {sr_i} Hz (expected {MIN_INPUT_SR}-{MAX_INPUT_SR} Hz)"
        )
    return sr_i


def _array_to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert any real numeric array to float32 in the nominal [-1, 1] range."""
    kind = arr.dtype.kind
    if kind == "f":
        return arr.astype(np.float32, copy=False)
    if kind == "i":
        scale = float(-np.iinfo(arr.dtype).min)  # 32768 for int16, 2**31 for int32
        return (arr.astype(np.float64) / scale).astype(np.float32)
    if kind == "u":
        half = (float(np.iinfo(arr.dtype).max) + 1.0) / 2.0  # 128 for uint8
        return ((arr.astype(np.float64) - half) / half).astype(np.float32)
    raise AudioLoadError(f"Unsupported array dtype {arr.dtype} (need int/uint/float)")


def _coerce_array(arr_like: Any) -> np.ndarray:
    """Return float32 array shaped (samples, channels)."""
    try:
        arr = np.asarray(arr_like)
    except Exception as exc:  # noqa: BLE001
        raise AudioLoadError(f"Cannot interpret input as an array: {exc}") from None
    arr = _array_to_float32(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2:
        # Accept (channels, samples) too, e.g. torch/librosa layout.
        if arr.shape[0] <= 8 < arr.shape[1]:
            arr = arr.T
    else:
        raise AudioLoadError(f"Audio array must be 1-D or 2-D, got {arr.ndim}-D")
    if arr.shape[1] > MAX_CHANNELS:
        raise AudioLoadError(
            f"Array has {arr.shape[1]} channels; expected shape (samples, channels)"
        )
    return arr


def _ffmpeg_decode(path: str) -> Tuple[np.ndarray, int]:
    """Decode *anything* ffmpeg understands to float32 WAV (native rate), then read it."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioLoadError(
            "This audio format needs ffmpeg, which is not installed "
            "(install ffmpeg or supply WAV/FLAC/OGG)."
        )
    with tempfile.TemporaryDirectory(prefix="svara_") as tmp:
        out = os.path.join(tmp, "decoded.wav")
        cmd = [exe, "-nostdin", "-v", "error", "-y", "-i", os.path.abspath(path),
               "-vn", "-f", "wav", "-c:a", "pcm_f32le", out]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise AudioLoadError("ffmpeg timed out while decoding the audio") from None
        if proc.returncode != 0 or not os.path.exists(out):
            msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            detail = msg[-1] if msg else "unknown error"
            raise AudioLoadError(f"Could not decode audio (not a valid audio file?): {detail}")
        try:
            data, sr = sf.read(out, dtype="float32", always_2d=True)
        except (sf.LibsndfileError, RuntimeError) as exc:
            raise AudioLoadError(f"Could not read ffmpeg output: {exc}") from None
    return data, int(sr)


def _check_duration_before_read(info_duration: float, max_duration_s: Optional[float]) -> None:
    if max_duration_s is not None and info_duration > max_duration_s:
        raise AudioLoadError(
            f"Audio is {info_duration:.1f} s long; the limit is {max_duration_s:.0f} s"
        )


def _soundfile_read(file_or_path: Any, max_duration_s: Optional[float]) -> Tuple[np.ndarray, int]:
    info = sf.info(file_or_path)  # cheap: header only; rejects non-audio quickly
    _check_duration_before_read(info.duration, max_duration_s)
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    data, sr = sf.read(file_or_path, dtype="float32", always_2d=True)
    return data, int(sr)


def _decode_path(path: Path, max_duration_s: Optional[float]) -> Tuple[np.ndarray, int, str]:
    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")
    if not path.is_file():
        raise AudioLoadError(f"Not a file: {path}")
    try:
        data, sr = _soundfile_read(str(path), max_duration_s)
        return data, sr, "soundfile"
    except AudioLoadError:
        raise
    except (sf.LibsndfileError, RuntimeError):
        data, sr = _ffmpeg_decode(str(path))
        return data, sr, "ffmpeg"


def _decode_bytes(raw: bytes, max_duration_s: Optional[float]) -> Tuple[np.ndarray, int, str]:
    if len(raw) == 0:
        raise AudioLoadError("Empty audio data (0 bytes)")
    try:
        data, sr = _soundfile_read(io.BytesIO(raw), max_duration_s)
        return data, sr, "soundfile"
    except AudioLoadError:
        raise
    except (sf.LibsndfileError, RuntimeError):
        with tempfile.TemporaryDirectory(prefix="svara_in_") as tmp:
            p = os.path.join(tmp, "input.bin")
            with open(p, "wb") as fh:
                fh.write(raw)
            data, sr = _ffmpeg_decode(p)
        return data, sr, "ffmpeg"


def _decode_source(
    source: Any, sample_rate: Optional[int], max_duration_s: Optional[float]
) -> Tuple[np.ndarray, int, str, str]:
    """Return (data[samples, channels] float32, sr, decoder, source_type)."""
    # ---- tuples: (array, sr) or (sr, array)
    if isinstance(source, tuple):
        if len(source) != 2:
            raise AudioLoadError("Tuple input must be (array, sample_rate) or (sample_rate, array)")
        a, b = source
        if isinstance(a, (int, float, np.integer, np.floating)) and not isinstance(
            b, (int, float, np.integer, np.floating)
        ):
            a, b = b, a
        return _coerce_array(a), _check_sample_rate(b), "array", "array"

    # ---- arrays (and lists)
    if isinstance(source, (np.ndarray, list)):
        if sample_rate is None:
            raise AudioLoadError("Array input needs sample_rate=... (or pass an (array, sr) tuple)")
        return _coerce_array(source), _check_sample_rate(sample_rate), "array", "array"

    # ---- paths
    if isinstance(source, (str, os.PathLike)):
        data, sr, dec = _decode_path(Path(source), max_duration_s)
        return data, sr, dec, "path"

    # ---- raw bytes
    if isinstance(source, (bytes, bytearray, memoryview)):
        data, sr, dec = _decode_bytes(bytes(source), max_duration_s)
        return data, sr, dec, "bytes"

    # ---- file-like (Streamlit UploadedFile, BytesIO, open file)
    if hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:  # noqa: BLE001
                pass
        raw = source.read()
    else:
        raise AudioLoadError(
            f"Unsupported input type {type(source).__name__}: pass a path, bytes, "
            "file-like object, or NumPy array"
        )
    if isinstance(raw, str):
        raise AudioLoadError("File-like object returned text; open it in binary mode ('rb')")
    data, sr, dec = _decode_bytes(bytes(raw), max_duration_s)
    return data, sr, dec, "file-like"


# --------------------------------------------------------------------------- processing
def resample_audio(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    """Band-limited resampling of a 1-D float signal.

    Uses libsoxr (``quality='HQ'``) when installed, otherwise
    ``scipy.signal.resample_poly`` with a Kaiser-windowed FIR low-pass.  Both include the
    anti-aliasing filter needed when lowering the sample rate (content above sr_out/2 is
    removed *before* decimation, not folded back into the speech band).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if sr_in == sr_out:
        return x
    if soxr is not None:
        y = soxr.resample(x, sr_in, sr_out, quality="HQ")
    else:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        y = resample_poly(x, int(sr_out) // g, int(sr_in) // g, window=("kaiser", 8.6))
    return np.asarray(y, dtype=np.float32)


def normalize_amplitude(
    x: np.ndarray,
    mode: Optional[str] = "peak",
    target_peak: float = DEFAULT_TARGET_PEAK,
    target_rms: float = DEFAULT_TARGET_RMS,
) -> Tuple[np.ndarray, float]:
    """Scale ``x`` to a consistent loudness.  Returns ``(y, linear_gain)``.

    * ``"peak"`` : largest |sample| becomes ``target_peak`` (default 0.9).
    * ``"rms"``  : RMS becomes ``target_rms`` (default 0.1 = -20 dBFS), but the gain is
      reduced if it would push the peak above 0.99 -- no hard clipping is ever introduced.
    * ``None`` / ``"none"`` : unchanged.

    Digital silence (peak < 1e-4) is returned untouched with gain 1.0, so a muted mic is
    not turned into amplified hiss.
    """
    if mode in (None, "none"):
        return x, 1.0
    if mode not in ("peak", "rms"):
        raise ValueError(f"Unknown normalisation mode {mode!r} (use 'peak', 'rms' or None)")
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target_peak must be in (0, 1]")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < SILENCE_PEAK:
        return x, 1.0
    if mode == "peak":
        gain = target_peak / peak
    else:
        if not 0.0 < target_rms < 1.0:
            raise ValueError("target_rms must be in (0, 1)")
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        gain = target_rms / max(rms, 1e-12)
        gain = min(gain, RMS_MODE_PEAK_LIMIT / peak)
    return (x * np.float32(gain)).astype(np.float32, copy=False), gain


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


# --------------------------------------------------------------------------- public API
def load_audio_with_report(
    source: Any,
    sample_rate: Optional[int] = None,
    *,
    target_sr: int = TARGET_SR,
    normalize: Optional[str] = "peak",
    target_peak: float = DEFAULT_TARGET_PEAK,
    target_rms: float = DEFAULT_TARGET_RMS,
    remove_dc: bool = True,
    max_duration_s: Optional[float] = 120.0,
    min_duration_s: float = 0.05,
    _source_type: Optional[str] = None,
) -> Tuple[np.ndarray, int, AudioReport]:
    """Load, condition and describe one clip.  See the module docstring.

    Parameters
    ----------
    source : path | bytes | file-like | ndarray | (array, sr)
    sample_rate : required only for bare NumPy arrays.
    target_sr : output rate (default 16 000 Hz).
    normalize : ``"peak"`` (default), ``"rms"`` or ``None``.
    remove_dc : subtract the mean (default True).
    max_duration_s : reject longer clips (``None`` disables).  A form answer is seconds long.
    min_duration_s : reject clips shorter than this after resampling.

    Returns
    -------
    signal : float32 ndarray, shape (n,), mono, values within [-1, 1]
    sr : int (== ``target_sr``)
    report : :class:`AudioReport`
    """
    target_sr = _check_sample_rate(target_sr, "target_sr")
    data, sr_in, decoder, source_type = _decode_source(source, sample_rate, max_duration_s)
    if _source_type:
        source_type = _source_type

    n_in, n_ch = data.shape
    if n_in == 0:
        raise AudioLoadError("The audio contains no samples")
    if not np.all(np.isfinite(data)):
        raise AudioLoadError("The audio contains NaN or infinite values")
    duration_in = n_in / sr_in
    if max_duration_s is not None and duration_in > max_duration_s:
        raise AudioLoadError(
            f"Audio is {duration_in:.1f} s long; the limit is {max_duration_s:.0f} s"
        )

    rep = AudioReport(
        source_type=source_type, decoder=decoder, original_sr=sr_in,
        original_channels=n_ch, original_duration_s=duration_in, output_sr=target_sr,
        normalization=normalize if normalize else "none",
    )
    rep.clipped_fraction_in = float(np.mean(np.abs(data) >= CLIP_LEVEL))
    if rep.clipped_fraction_in > CLIP_WARN_FRACTION:
        rep.warnings.append(
            f"{rep.clipped_fraction_in * 100:.2f} % of samples are clipped; "
            "the recording was too loud (lower the mic gain)."
        )

    # 1) mono: average the channels
    x = data[:, 0].copy() if n_ch == 1 else data.mean(axis=1, dtype=np.float32)

    # 2) DC removal
    if remove_dc:
        dc = float(np.mean(x, dtype=np.float64))
        x = (x - np.float32(dc)).astype(np.float32, copy=False)
        rep.dc_offset_removed = dc

    # 3) resample (anti-aliased)
    if sr_in != target_sr:
        x = resample_audio(x, sr_in, target_sr)
        rep.resampled = True
    if x.size == 0 or x.size / target_sr < min_duration_s:
        raise AudioLoadError(
            f"Audio is too short ({x.size / target_sr * 1000:.0f} ms); "
            f"need at least {min_duration_s * 1000:.0f} ms"
        )

    # 4) amplitude normalisation
    rep.peak_before = float(np.max(np.abs(x)))
    rep.rms_before = _rms(x)
    rep.is_silent = rep.peak_before < SILENCE_PEAK
    x, gain = normalize_amplitude(x, normalize, target_peak, target_rms)
    # Resampling filters can overshoot slightly; keep strictly inside [-1, 1].
    np.clip(x, -1.0, 1.0, out=x)
    x = np.ascontiguousarray(x, dtype=np.float32)

    rep.gain_applied = float(gain)
    rep.gain_db = 20.0 * math.log10(gain) if gain > 0 else 0.0
    rep.peak_after = float(np.max(np.abs(x)))
    rep.rms_after = _rms(x)
    rep.output_duration_s = x.size / target_sr
    if rep.is_silent:
        rep.warnings.append("The recording is silent (no signal detected). Check the microphone.")
    elif rep.gain_db > HIGH_GAIN_WARN_DB:
        rep.warnings.append(
            f"Input was very quiet; it was amplified by {rep.gain_db:.0f} dB, "
            "so background noise may be loud. Speak closer to the microphone."
        )
    for w in rep.warnings:
        log.warning(w)
    return x, target_sr, rep


def load_audio(source: Any, sample_rate: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, int]:
    """The 1A contract: ``signal, sr = load_audio(source)`` (mono, 16 kHz, normalised).

    Keyword arguments are those of :func:`load_audio_with_report`.
    """
    signal, sr, _ = load_audio_with_report(source, sample_rate, **kwargs)
    return signal, sr


def record_from_mic(
    duration_s: float = 5.0,
    device: Optional[Union[int, str]] = None,
    **load_kwargs: Any,
) -> Tuple[np.ndarray, int]:
    """Record ``duration_s`` seconds from the microphone and return ``(signal, 16000)``.

    Needs the optional ``sounddevice`` package (and PortAudio).  Recording is requested
    at 16 kHz mono; if the device refuses that rate it is opened at its native rate and the
    shared pipeline resamples.  For web apps prefer ``st.audio_input`` and pass its
    object to :func:`load_audio` -- the browser owns the microphone there.
    """
    signal, sr, _ = record_from_mic_with_report(duration_s, device, **load_kwargs)
    return signal, sr


def record_from_mic_with_report(
    duration_s: float = 5.0,
    device: Optional[Union[int, str]] = None,
    **load_kwargs: Any,
) -> Tuple[np.ndarray, int, AudioReport]:
    if not (isinstance(duration_s, (int, float)) and 0 < duration_s <= 600):
        raise AudioLoadError("duration_s must be a number between 0 and 600 seconds")
    try:
        import sounddevice as sd
    except Exception as exc:  # ImportError, or OSError when PortAudio is missing
        raise AudioLoadError(
            "Microphone recording needs the 'sounddevice' package and PortAudio "
            f"({exc}). Alternatively load a file / use st.audio_input."
        ) from None

    def _rec(rate: int) -> np.ndarray:
        frames = int(round(duration_s * rate))
        buf = sd.rec(frames, samplerate=rate, channels=1, dtype="float32", device=device)
        sd.wait()
        return np.asarray(buf, dtype=np.float32)

    rate = TARGET_SR
    try:
        rec = _rec(rate)
    except Exception:  # device does not support 16 kHz -> use its default rate
        try:
            dev_info = sd.query_devices(device, "input")
            rate = int(dev_info["default_samplerate"])
            rec = _rec(rate)
        except Exception as exc:  # noqa: BLE001
            raise AudioLoadError(f"Could not record from the microphone: {exc}") from None
    load_kwargs.setdefault("max_duration_s", None)
    return load_audio_with_report(
        (rec, rate), _source_type="microphone", **load_kwargs
    )


def save_wav(signal: np.ndarray, path: Union[str, os.PathLike], sr: int = TARGET_SR) -> str:
    """Write a mono float signal to a 16-bit PCM WAV file (clipped to [-1, 1])."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise AudioLoadError("save_wav expects a 1-D (mono) signal")
    if not np.all(np.isfinite(x)):
        raise AudioLoadError("Cannot save NaN/Inf samples")
    sf.write(str(path), np.clip(x, -1.0, 1.0), int(sr), subtype="PCM_16")
    return str(path)


# --------------------------------------------------------------------------- CLI
def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse

    p = argparse.ArgumentParser(description="Convert audio to 16 kHz mono, normalised WAV.")
    p.add_argument("input")
    p.add_argument("output", nargs="?", help="optional output .wav path")
    p.add_argument("--normalize", choices=["peak", "rms", "none"], default="peak")
    args = p.parse_args(argv)
    try:
        sig, sr, rep = load_audio_with_report(
            args.input, normalize=None if args.normalize == "none" else args.normalize
        )
    except AudioLoadError as exc:
        print(f"error: {exc}")
        return 1
    print(rep)
    if args.output:
        print("saved:", save_wav(sig, args.output, sr))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
