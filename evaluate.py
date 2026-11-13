"""
evaluate.py  --  SVARA, Member 1 / Sub-module 1C: Noise-robustness experiment
=============================================================================

Does the 1B preprocessing chain help the speech recogniser when the input is noisy?

    python evaluate.py --clips data/test_clips --refs data/test_clips/references.csv
    python evaluate.py --clips data/test_clips_tts --model base.en         # stand-in clips
    python evaluate.py ... --set low_hz=100 --set denoise=none             # 1B tuning run

For every clip, noise kind and SNR (default: clean, 20, 10, 5 dB white noise):

1. load the clip with 1A (``load_audio``: mono, 16 kHz, peak 0.9);
2. add noise at the requested SNR (:func:`add_noise`; SNR measured against the power of
   the *speech* part, found with 1B's VAD on the clean clip, not of the whole file);
3. transcribe it twice -- **raw** (1A only) and **preprocessed** (``clean_audio``);
4. score each transcript against the reference with the word error rate
   ``WER = (S + D + I) / N`` (:func:`word_errors`, a word-level Levenshtein alignment).

Outputs (``--out``, default ``outputs/eval``): ``per_clip.csv`` (every transcript and its
errors), ``summary.csv`` (corpus WER per SNR and condition), ``wer_vs_snr.png`` and a
Markdown table printed to the console for the README.

The recogniser is pluggable: ``--asr auto`` uses Member 2's ``asr_agent.transcribe`` when
that module exists, otherwise a local OpenAI Whisper model (``--model``).  Any callable
``transcribe(signal, language) -> str`` can be passed to :func:`run_experiment`.

References CSV columns: ``file,transcript`` and optionally ``language`` (en/hi/ml/kn).
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from audio_io import TARGET_SR, load_audio
from preprocess import PreprocessConfig, clean_audio, detect_speech

log = logging.getLogger(__name__)

__all__ = [
    "Clip", "normalize_text", "word_errors", "corpus_wer", "make_noise", "speech_power",
    "add_noise", "measured_snr", "load_clips", "get_transcriber", "run_experiment",
    "summarize", "write_csv", "markdown_table",
]

DEFAULT_SNRS: Tuple[Union[str, float], ...] = ("clean", 20, 10, 5)
Transcriber = Callable[[np.ndarray, Optional[str]], str]


# =========================================================================== WER
def normalize_text(s: str, language: Optional[str] = None) -> str:
    """Make two transcripts comparable before counting word errors.

    Always: lower-case, drop punctuation and symbols, collapse spaces.  This works for
    Devanagari / Malayalam / Kannada too: only Unicode *punctuation* (P*) and *symbol* (S*)
    characters are removed, so vowel signs and viramas (category M*) stay.

    English (``language="en"``, or no language and pure-ASCII text) additionally goes
    through Whisper's ``EnglishTextNormalizer`` when it is installed -- the normaliser used
    for the WER figures in the Whisper paper.  It writes numbers as digits ("twenty five"
    and "25" both become "25", "fifteenth" -> "15th", "nineteen eighty" -> "1980"), which
    matters here because form answers are full of ages, dates and phone numbers: without
    it a perfectly recognised "25" would count as two word errors.  Neighbouring digit
    groups are then joined ("98472 16530" -> "9847216530"), as a phone number is one value.
    """
    s = unicodedata.normalize("NFC", s)
    english = language == "en" or (language is None and s.isascii())
    if english and _english_normalizer() is not None:
        s = _english_normalizer()(s)
        return re.sub(r"(?<=\d) (?=\d)", "", s).strip()
    s = s.casefold().replace("-", " ")
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] not in "PS")
    return re.sub(r"\s+", " ", s).strip()


_EN_NORMALIZER: List[Any] = []


def _english_normalizer():
    if not _EN_NORMALIZER:
        try:
            from whisper.normalizers import EnglishTextNormalizer
            _EN_NORMALIZER.append(EnglishTextNormalizer())
        except ImportError:
            _EN_NORMALIZER.append(None)
    return _EN_NORMALIZER[0]


def word_errors(reference: str, hypothesis: str, normalize: bool = True,
                language: Optional[str] = None) -> Dict[str, Any]:
    """Align two word sequences with minimum edit distance (Levenshtein, all costs 1) and
    count substitutions S, deletions D (reference words missing) and insertions I (extra
    words).  ``wer = (S + D + I) / N`` with ``N`` = number of reference words; it can
    exceed 1 when the recogniser adds many words.  An empty reference gives
    ``wer = 0`` if the hypothesis is empty too, else ``1``.
    """
    if normalize:
        reference = normalize_text(reference, language)
        hypothesis = normalize_text(hypothesis, language)
    r, h = reference.split(), hypothesis.split()
    n, m = len(r), len(h)
    # dp[i][j] = edit distance between r[:i] and h[:j]
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1, j - 1] + (r[i - 1] != h[j - 1])
            dp[i, j] = min(sub, dp[i - 1, j] + 1, dp[i, j - 1] + 1)
    # back-trace to split the distance into S / D / I
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += r[i - 1] != h[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    errors = S + D + I
    wer = errors / n if n else float(m > 0)
    return {"S": int(S), "D": int(D), "I": int(I), "N": n, "errors": int(errors), "wer": wer}


def corpus_wer(results: Iterable[Dict[str, Any]]) -> float:
    """Total errors / total reference words (the standard corpus-level WER; long answers
    weigh more than short ones, unlike the mean of per-clip WERs)."""
    rows = list(results)
    n = sum(r["N"] for r in rows)
    e = sum(r["errors"] for r in rows)
    return e / n if n else 0.0


# =========================================================================== noise
def make_noise(kind: str, n: int, seed: int = 0) -> np.ndarray:
    """Unit-variance noise: ``white`` (flat spectrum, like hiss) or ``pink`` (power ~ 1/f,
    like fans and room rumble)."""
    rng = np.random.default_rng(seed)
    if kind == "white":
        y = rng.standard_normal(n)
    elif kind == "pink":
        X = np.fft.rfft(rng.standard_normal(n))
        f = np.arange(X.size, dtype=np.float64)
        f[0] = 1.0
        y = np.fft.irfft(X / np.sqrt(f), n)
    else:
        raise ValueError(f"unknown noise kind {kind!r} (use 'white' or 'pink')")
    return y / max(float(np.std(y)), 1e-12)


def speech_power(x: np.ndarray, sr: int = TARGET_SR) -> float:
    """Mean power of the *speech* samples of a clean clip (1B's VAD, without padding).
    Falls back to the whole clip when no speech can be found."""
    x = np.asarray(x, dtype=np.float64)
    vad = detect_speech(x.astype(np.float32), sr)
    if vad.reliable and vad.speech_detected:
        idx = np.concatenate([np.arange(s, e) for s, e in vad.segments])
        return float(np.mean(x[idx] ** 2))
    return float(np.mean(x ** 2))


def add_noise(clean: np.ndarray, snr_db: float, kind: str = "white", seed: int = 0,
              sr: int = TARGET_SR) -> np.ndarray:
    """Mix noise into ``clean`` so that ``10 log10(P_speech / P_noise) = snr_db``.

    ``P_speech`` is measured on the speech frames only (:func:`speech_power`): measured on
    the whole file, the pauses would lower it and the real SNR during the words would be
    better than claimed.  If the mix would exceed full scale, the whole mix is scaled down
    (that keeps the SNR).  Returns float32.
    """
    x = np.asarray(clean, dtype=np.float64)
    noise = make_noise(kind, x.size, seed) * np.sqrt(speech_power(x, sr) / 10 ** (snr_db / 10))
    y = x + noise
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y /= peak
    return y.astype(np.float32)


def measured_snr(clean: np.ndarray, noisy: np.ndarray, sr: int = TARGET_SR) -> float:
    """SNR of a mix, measured the same way :func:`add_noise` defines it (scale-invariant)."""
    c = np.asarray(clean, np.float64)
    y = np.asarray(noisy, np.float64)
    g = np.dot(y, c) / np.dot(c, c)                 # undo a possible rescale of the mix
    return 10 * np.log10(speech_power(c, sr) * g * g / np.mean((y - g * c) ** 2))


# =========================================================================== data
@dataclass
class Clip:
    name: str
    signal: np.ndarray            # 1A output: mono, 16 kHz, peak 0.9
    reference: str
    language: Optional[str] = None


def load_clips(clips_dir: str, refs_csv: Optional[str] = None,
               limit: Optional[int] = None) -> List[Clip]:
    """Read ``refs_csv`` (default ``<clips_dir>/references.csv``) and load every listed file
    with 1A.  Missing files are an error (a silently smaller test set is misleading)."""
    refs_csv = refs_csv or os.path.join(clips_dir, "references.csv")
    with open(refs_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not {"file", "transcript"} <= set(rows[0]):
        raise ValueError(f"{refs_csv} needs the columns 'file' and 'transcript'")
    missing = [r["file"] for r in rows if not os.path.isfile(os.path.join(clips_dir, r["file"]))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clip(s) listed in {refs_csv} not found in "
                                f"{clips_dir}: {', '.join(missing[:5])}")
    clips = []
    for r in rows[:limit]:
        x, _ = load_audio(os.path.join(clips_dir, r["file"]))
        clips.append(Clip(r["file"], x, r["transcript"], (r.get("language") or "").strip() or None))
    return clips


# =========================================================================== ASR
def get_transcriber(asr: str = "auto", model: str = "small", device: Optional[str] = None
                    ) -> Tuple[Transcriber, str]:
    """Return ``(transcribe(signal, language) -> text, description)``.

    * ``"asr_agent"`` -- Member 2's ``asr_agent.transcribe`` (may return text or
      ``(text, language)``).
    * ``"whisper"``   -- local OpenAI Whisper ``model`` (tiny/base/small/..., ``.en`` for
      English-only), greedy decoding (temperature 0) so runs are repeatable.
    * ``"auto"``      -- ``asr_agent`` if it can be imported, else Whisper.
    """
    if asr in ("auto", "asr_agent"):
        try:
            import asr_agent
            fn = asr_agent.transcribe

            def via_agent(x, language=None):
                try:
                    out = fn(x, language=language) if language else fn(x)
                except TypeError:
                    out = fn(x)
                return out[0] if isinstance(out, tuple) else str(out)
            return via_agent, "asr_agent.transcribe"
        except ImportError:
            if asr == "asr_agent":
                raise
    if asr not in ("auto", "whisper"):
        raise ValueError(f"unknown ASR backend {asr!r}")
    try:
        import whisper
    except ImportError:
        raise ImportError("the Whisper backend needs 'openai-whisper' (pip install openai-whisper)"
                          ) from None
    m = whisper.load_model(model, device=device)
    english_only = model.endswith(".en")

    def via_whisper(x, language=None):
        lang = "en" if english_only else language
        out = m.transcribe(np.asarray(x, np.float32), language=lang, task="transcribe",
                           temperature=0.0, fp16=False, condition_on_previous_text=False,
                           verbose=None)
        return out["text"].strip()
    return via_whisper, f"whisper-{model}"


# =========================================================================== experiment
def _snr_key(s) -> Union[str, float]:
    return "clean" if str(s).lower() == "clean" else float(s)


def run_experiment(clips: Sequence[Clip], transcribe: Transcriber,
                   snrs: Sequence[Union[str, float]] = DEFAULT_SNRS,
                   noise_kinds: Sequence[str] = ("white",), seeds: Sequence[int] = (0,),
                   preprocess: Optional[Dict[str, Any]] = None,
                   progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Transcribe every clip x noise kind x SNR x seed, raw and preprocessed.

    ``preprocess`` holds :class:`preprocess.PreprocessConfig` overrides for the
    "preprocessed" condition (that is how 1B tunes its settings).  Returns one row per
    transcription with the reference, hypothesis, S/D/I/N and WER.
    """
    cfg_over = dict(preprocess or {})
    PreprocessConfig(**cfg_over).validate(TARGET_SR)     # fail early on a bad override
    rows: List[Dict[str, Any]] = []
    snrs = [_snr_key(s) for s in snrs]
    # the clean condition does not depend on the noise kind or seed: run it once
    trials = [("none", "clean", seeds[0])] if "clean" in snrs else []
    trials += [(k, s, sd) for k in noise_kinds for s in snrs if s != "clean" for sd in seeds]
    for ci, clip in enumerate(clips):
        for kind, snr, seed in trials:
            if snr == "clean":
                noisy = clip.signal
            else:
                # a different noise realisation per clip, the same one for raw/preprocessed
                noisy = add_noise(clip.signal, snr, kind, seed=1000 * seed + ci)
            inputs = {
                "raw": load_audio(noisy, TARGET_SR)[0],          # what 1A would pass on
                "preprocessed": clean_audio(noisy, TARGET_SR, **cfg_over),
            }
            for cond, sig in inputs.items():
                t = time.perf_counter()
                hyp = transcribe(sig, clip.language)
                dt = time.perf_counter() - t
                e = word_errors(clip.reference, hyp, language=clip.language)
                rows.append({"clip": clip.name, "language": clip.language or "",
                             "noise": kind, "snr": snr,
                             "seed": seed, "condition": cond,
                             "duration_s": round(len(sig) / TARGET_SR, 3),
                             "reference": clip.reference, "hypothesis": hyp,
                             **e, "asr_s": round(dt, 3)})
                if progress:
                    progress(f"[{clip.name} | "
                             f"{'clean' if snr == 'clean' else f'{kind} {snr:g} dB'} | {cond}] "
                             f"WER {e['wer']:.2f}  '{hyp}'")
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Corpus WER per (noise, SNR, condition), plus the spread over clips (``wer_std``)
    and the standard error of the mean per-clip WER (``wer_se``, drawn as error bars).

    The clean rows (``noise="none"``) are repeated under every noise kind so that each
    curve starts from the same clean point.
    """
    kinds = sorted({r["noise"] for r in rows if r["noise"] != "none"}) or ["none"]
    out = []
    for kind in kinds:
        for snr in dict.fromkeys(r["snr"] for r in rows):
            for cond in dict.fromkeys(r["condition"] for r in rows):
                sel = [r for r in rows if r["snr"] == snr and r["condition"] == cond
                       and (r["noise"] == kind or snr == "clean")]
                if not sel:
                    continue
                per_clip = [r["wer"] for r in sel]
                out.append({"noise": kind, "snr": snr, "condition": cond,
                            "wer": corpus_wer(sel), "wer_mean": float(np.mean(per_clip)),
                            "wer_std": float(np.std(per_clip)),
                            "wer_se": float(np.std(per_clip, ddof=1) / np.sqrt(len(sel)))
                            if len(sel) > 1 else 0.0, "n": len(sel),
                            "words": sum(r["N"] for r in sel),
                            "S": sum(r["S"] for r in sel), "D": sum(r["D"] for r in sel),
                            "I": sum(r["I"] for r in sel),
                            "exact": sum(r["errors"] == 0 for r in sel)})
    return out


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def markdown_table(summary: Sequence[Dict[str, Any]]) -> str:
    """README-style table: one row per SNR, WER without / with preprocessing."""
    lines = []
    for kind in dict.fromkeys(s["noise"] for s in summary):
        sel = [s for s in summary if s["noise"] == kind]
        lines += [f"**{kind} noise**", "",
                  "| SNR | WER (no preprocessing) | WER (with preprocessing) |",
                  "|---|---|---|"]
        for snr in dict.fromkeys(s["snr"] for s in sel):
            w = {s["condition"]: s["wer"] for s in sel if s["snr"] == snr}
            fmt = lambda c: f"{100 * w[c]:.1f} %" if c in w else "-"
            lab = "Clean" if snr == "clean" else f"{snr:g} dB"
            lines.append(f"| {lab} | {fmt('raw')} | {fmt('preprocessed')} |")
        lines.append("")
    return "\n".join(lines)


def plot_summary(summary: Sequence[Dict[str, Any]], path: str, title_extra: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import plot_wer_vs_snr

    kinds = list(dict.fromkeys(s["noise"] for s in summary))
    fig, ax = plt.subplots(1, len(kinds), figsize=(6.2 * len(kinds), 4), squeeze=False)
    for a, kind in zip(ax[0], kinds):
        plot_wer_vs_snr([s for s in summary if s["noise"] == kind], a,
                        title=f"WER vs SNR, {kind} noise{title_extra}")
    ymax = max(a.get_ylim()[1] for a in ax[0])
    for a in ax[0]:
        a.set_ylim(0, ymax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =========================================================================== CLI
def _parse_value(v: str) -> Any:
    low = v.lower()
    if low in ("none", "null", "off"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse

    p = argparse.ArgumentParser(description="SVARA 1C: WER vs SNR with and without preprocessing.")
    p.add_argument("--clips", default=os.path.join("data", "test_clips"), help="folder of clips")
    p.add_argument("--refs", help="references CSV (default: <clips>/references.csv)")
    p.add_argument("--snr", nargs="+", default=[str(s) for s in DEFAULT_SNRS],
                   help="SNRs in dB, plus 'clean' (default: clean 20 10 5)")
    p.add_argument("--noise", nargs="+", default=["white"], choices=["white", "pink"])
    p.add_argument("--seeds", type=int, default=1, help="noise realisations per clip")
    p.add_argument("--asr", default="auto", choices=["auto", "whisper", "asr_agent"])
    p.add_argument("--model", default="small", help="Whisper model (tiny, base, small, base.en, ...)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="PreprocessConfig override for the preprocessed condition, e.g. "
                        "--set low_hz=100 --set denoise=none (repeatable)")
    p.add_argument("--limit", type=int, help="use only the first N clips (quick check)")
    p.add_argument("--out", default=os.path.join("outputs", "eval"))
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    over = {}
    for item in a.set:
        k, sep, v = item.partition("=")
        if not sep or k not in PreprocessConfig.__dataclass_fields__:
            p.error(f"--set needs FIELD=VALUE with a PreprocessConfig field, got {item!r}")
        over[k] = _parse_value(v)
    clips = load_clips(a.clips, a.refs, a.limit)
    transcribe, asr_name = get_transcriber(a.asr, a.model)
    print(f"{len(clips)} clips, ASR = {asr_name}, noise = {a.noise}, SNR = {a.snr}, "
          f"seeds = {a.seeds}, preprocessing overrides = {over or 'none (defaults)'}")
    t = time.perf_counter()
    rows = run_experiment(clips, transcribe, a.snr, a.noise, range(a.seeds), over,
                          progress=None if a.quiet else print)
    summary = summarize(rows)
    write_csv(rows, os.path.join(a.out, "per_clip.csv"))
    write_csv(summary, os.path.join(a.out, "summary.csv"))
    png = plot_summary(summary, os.path.join(a.out, "wer_vs_snr.png"), f" ({asr_name})")
    print(f"\n{len(rows)} transcriptions in {time.perf_counter() - t:.0f} s\n")
    print(markdown_table(summary))
    print("written:", os.path.join(a.out, "per_clip.csv"), os.path.join(a.out, "summary.csv"), png)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
