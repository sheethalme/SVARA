"""
plots.py  --  SVARA, Member 1 / Sub-module 1C: Visualisation (Signal Lab)
=========================================================================

Matplotlib figures for one recording, before vs after the 1B preprocessing chain::

    from preprocess import clean_audio_with_stages
    from plots import plot_before_after

    result = clean_audio_with_stages("answer.wav")
    fig = plot_before_after(result)            # waveform, spectrogram, MFCC, pitch
    fig.savefig("signal_lab.png")              # or st.pyplot(fig) in the app

Single panels (each draws into ``ax=`` or a new figure and returns the ``Axes``):
:func:`plot_waveform`, :func:`plot_spectrogram`, :func:`plot_mel_spectrogram`,
:func:`plot_mfcc`, :func:`plot_pitch`, :func:`plot_time_features`, :func:`plot_vad`.
Analytics: :func:`plot_wer_vs_snr` (used by ``evaluate.py``).

Rules followed by every before/after figure
-------------------------------------------
* **Same time axis**: the trimmed output is drawn at its original position in the clip
  (shifted by the VAD trim start), so a sound sits at the same x position in both columns.
* **Same colour scale for the spectrograms**: both use limits taken from the *input*, so
  "darker" really means "less energy", not a different colour range.  The MFCC maps each
  get their own symmetric scale instead: pre-emphasis and the band-pass shift the average
  of c1-c6 so much that a shared scale would saturate one of the two columns.
* The module never calls ``matplotlib.use``; scripts choose the backend (``Agg`` for files).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from audio_io import TARGET_SR
from features import _n_frames, extract_features, frame_signal, power_spectrogram

__all__ = [
    "COLORS", "plot_waveform", "plot_spectrogram", "plot_mel_spectrogram", "plot_mfcc",
    "plot_pitch", "plot_time_features", "plot_vad", "plot_before_after",
    "plot_feature_overview", "plot_wer_vs_snr",
]

COLORS = {"blue": "#2a6fbb", "orange": "#d9822b", "green": "#2e8b57", "red": "#c0392b",
          "grey": "#777777", "purple": "#7b4ea3"}
SPEC_CMAP = "magma"
SPEC_RANGE_DB = 80.0          # spectrograms show the top 80 dB below the loudest bin


def _ax(ax, figsize=(8, 2.6)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _feats(x, sr, feats):
    return feats if feats is not None else extract_features(np.asarray(x, np.float32), sr)


# =========================================================================== single panels
def plot_waveform(x: np.ndarray, sr: int = TARGET_SR, ax=None, *, offset_s: float = 0.0,
                  title: str = "Waveform", color: str = COLORS["blue"],
                  xlim: Optional[Sequence[float]] = None):
    """Amplitude against time.  ``offset_s`` places a trimmed clip at its original time."""
    ax = _ax(ax)
    t = np.arange(len(x)) / sr + offset_s
    ax.plot(t, x, lw=0.4, color=color)
    ax.set(title=title, xlabel="time (s)", ylabel="amplitude", ylim=(-1.05, 1.05))
    if xlim is not None:
        ax.set_xlim(*xlim)
    return ax


def plot_spectrogram(x: np.ndarray, sr: int = TARGET_SR, ax=None, *, offset_s: float = 0.0,
                     title: str = "Spectrogram", vmin: Optional[float] = None,
                     vmax: Optional[float] = None, n_fft: int = 512, hop: int = 160,
                     frame_len: int = 400, xlim: Optional[Sequence[float]] = None,
                     colorbar: bool = False):
    """Short-time power spectrum (Hamming window, 25 ms / 10 ms) in dB.  Returns the
    ``Axes``; the image is ``ax.images[0]``/``ax.collections[0]``."""
    ax = _ax(ax)
    x = np.asarray(x, np.float64)
    T = _n_frames(x.size, frame_len, hop)
    P = power_spectrogram(frame_signal(x, frame_len, hop, T), n_fft)
    S = 10 * np.log10(P + 1e-12)
    vmax = float(S.max()) if vmax is None else vmax
    vmin = vmax - SPEC_RANGE_DB if vmin is None else vmin
    t = (np.arange(T) * hop + frame_len / 2) / sr + offset_s
    f = np.linspace(0, sr / 2, P.shape[0])
    m = ax.pcolormesh(t, f, S, vmin=vmin, vmax=vmax, shading="auto", cmap=SPEC_CMAP)
    ax.set(title=title, xlabel="time (s)", ylabel="frequency (Hz)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if colorbar:
        ax.figure.colorbar(m, ax=ax, label="dB")
    return ax


def plot_mel_spectrogram(feats: Dict[str, Any], ax=None, *, offset_s: float = 0.0,
                         title: str = "Mel spectrogram", vmin=None, vmax=None,
                         xlim=None, colorbar: bool = False):
    """Energy in the mel bands (dB).  The y axis is labelled with band centres in Hz."""
    ax = _ax(ax)
    M = feats["mel_spectrogram_db"]
    vmax = float(M.max()) if vmax is None else vmax
    vmin = vmax - SPEC_RANGE_DB if vmin is None else vmin
    t = feats["times"] + offset_s
    m = ax.pcolormesh(t, np.arange(M.shape[0]), M, vmin=vmin, vmax=vmax, shading="auto",
                      cmap=SPEC_CMAP)
    idx = np.linspace(0, M.shape[0] - 1, 5).round().astype(int)
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{feats['mel_frequencies'][i]:.0f}" for i in idx])
    ax.set(title=title, xlabel="time (s)", ylabel="mel band (Hz)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if colorbar:
        ax.figure.colorbar(m, ax=ax, label="dB")
    return ax


def plot_mfcc(feats: Dict[str, Any], ax=None, *, offset_s: float = 0.0, title: str = "MFCC",
              vlim: Optional[float] = None, skip_c0: bool = True, xlim=None,
              colorbar: bool = False):
    """MFCC heat-map (coefficient against time).  ``c0`` (log energy) is hidden by default
    because its large range would wash out the colours of c1-c12."""
    ax = _ax(ax)
    C = feats["mfcc"][1:] if skip_c0 else feats["mfcc"]
    first = 1 if skip_c0 else 0
    vlim = float(np.percentile(np.abs(C), 99)) if vlim is None else vlim
    t = feats["times"] + offset_s
    m = ax.pcolormesh(t, np.arange(first, first + C.shape[0]), C, vmin=-vlim, vmax=vlim,
                      shading="auto", cmap="coolwarm")
    ax.set(title=title, xlabel="time (s)", ylabel="coefficient")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if colorbar:
        ax.figure.colorbar(m, ax=ax)
    return ax


def plot_pitch(feats: Dict[str, Any], ax=None, *, offset_s: float = 0.0,
               title: str = "Pitch contour (F0)", color: str = COLORS["orange"],
               energy: bool = True, ylim: Optional[Sequence[float]] = None, xlim=None):
    """F0 in Hz (gaps where unvoiced), with the short-time energy in dB as a grey line on a
    second axis -- pitch only exists where there is voiced energy."""
    ax = _ax(ax)
    t = feats["times"] + offset_s
    ax.plot(t, feats["f0"], ".", ms=2.5, color=color, label="F0")
    f0 = feats["f0"][feats["voiced"]]
    if ylim is None:
        ylim = (0, max(300.0, 1.2 * float(np.max(f0)) if f0.size else 300.0))
    ax.set(title=title, xlabel="time (s)", ylabel="F0 (Hz)", ylim=ylim)
    if energy:
        ax2 = ax.twinx()
        ax2.plot(t, feats["energy_db"], lw=0.7, color=COLORS["grey"], alpha=0.7)
        ax2.set_ylabel("energy (dB)", color=COLORS["grey"])
        ax2.tick_params(axis="y", colors=COLORS["grey"])
        ax2.spines["right"].set_visible(True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return ax


def plot_time_features(feats: Dict[str, Any], axes=None, *, offset_s: float = 0.0, xlim=None):
    """Three stacked panels: short-time energy (dB), zero-crossing rate, spectral centroid."""
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(8, 5.4), sharex=True)
    t = feats["times"] + offset_s
    for a, key, lab, col in [(axes[0], "energy_db", "energy (dB)", COLORS["blue"]),
                             (axes[1], "zcr", "ZCR", COLORS["green"]),
                             (axes[2], "spectral_centroid", "centroid (Hz)", COLORS["purple"])]:
        a.plot(t, feats[key], lw=0.9, color=col)
        a.set_ylabel(lab)
        if xlim is not None:
            a.set_xlim(*xlim)
    axes[0].set_title("Short-time energy, zero-crossing rate, spectral centroid")
    axes[-1].set_xlabel("time (s)")
    return axes


def plot_vad(vad, ax=None, *, trim=None, sr: int = TARGET_SR, title: str =
             "Voice activity detection"):
    """1B's VAD decision: frame energy, threshold and the frames classed as speech.
    ``trim=(start_sample, end_sample)`` marks the part that was kept."""
    ax = _ax(ax)
    ax.plot(vad.frame_times_s, vad.energy_db, lw=0.9, color=COLORS["blue"], label="energy")
    ax.axhline(vad.threshold_db, color=COLORS["red"], ls="--", lw=1, label="threshold")
    lo, hi = float(vad.energy_db.min()), float(vad.energy_db.max())
    ax.fill_between(vad.frame_times_s, lo, hi, where=vad.mask, color=COLORS["green"],
                    alpha=0.18, label="speech")
    if trim is not None:
        ax.axvline(trim[0] / sr, color="k", lw=0.8)
        ax.axvline(trim[1] / sr, color="k", lw=0.8, label="kept")
    ax.set(title=title, xlabel="time (s)", ylabel="energy (dB)")
    ax.legend(fontsize=7, loc="lower right")
    return ax


# =========================================================================== figures
def plot_before_after(before: Any, after: Optional[np.ndarray] = None, sr: int = TARGET_SR,
                      *, offset_s: float = 0.0, titles=("Before preprocessing",
                                                        "After preprocessing"),
                      feats_before=None, feats_after=None, figsize=(11, 9.5)):
    """The Signal Lab figure: 4 rows (waveform, spectrogram, MFCC heat-map, pitch contour)
    x 2 columns (before, after).

    Pass a :class:`preprocess.PreprocessResult` as ``before`` (then input, output and the
    trim offset are taken from it), or two arrays plus ``offset_s`` = where ``after``
    starts inside ``before`` (in seconds).
    """
    if after is None:                                   # a PreprocessResult
        res = before
        before, after, sr = res.stages["input"], res.signal, res.sr
        offset_s = res.trim_start / res.sr
    before = np.asarray(before, np.float32)
    after = np.asarray(after, np.float32)
    fb = _feats(before, sr, feats_before)
    fa = _feats(after, sr, feats_after)
    xlim = (0.0, max(len(before) / sr, offset_s + len(after) / sr))

    fig, ax = plt.subplots(4, 2, figsize=figsize, sharex=True)
    # shared colour limits, from the input
    plot_spectrogram(before, sr, ax[1, 0], title="Spectrogram", xlim=xlim)
    vmin, vmax = ax[1, 0].collections[0].get_clim()
    plot_spectrogram(after, sr, ax[1, 1], offset_s=offset_s, title="Spectrogram",
                     vmin=vmin, vmax=vmax, xlim=xlim)
    f0_all = np.concatenate([fb["f0"][fb["voiced"]], fa["f0"][fa["voiced"]]])
    ylim = (0, max(300.0, 1.2 * float(f0_all.max()) if f0_all.size else 300.0))
    for j, (x, f, off) in enumerate([(before, fb, 0.0), (after, fa, offset_s)]):
        col = COLORS["blue"] if j == 0 else COLORS["green"]
        plot_waveform(x, sr, ax[0, j], offset_s=off, title=titles[j], color=col, xlim=xlim)
        plot_mfcc(f, ax[2, j], offset_s=off, title="MFCC (c1-c12)", xlim=xlim)
        plot_pitch(f, ax[3, j], offset_s=off, ylim=ylim, xlim=xlim)
    for a in ax[:-1].ravel():
        a.set_xlabel("")
    if offset_s > 0 or offset_s + len(after) / sr < len(before) / sr:
        for a in ax[:, 1]:
            a.axvspan(xlim[0], offset_s, color=COLORS["grey"], alpha=0.12, lw=0)
            a.axvspan(offset_s + len(after) / sr, xlim[1], color=COLORS["grey"], alpha=0.12, lw=0)
    fig.tight_layout()
    return fig


def plot_feature_overview(x: np.ndarray, sr: int = TARGET_SR, feats=None, *,
                          title: str = "Features", figsize=(10, 10)):
    """Every 1C feature of one signal: waveform, spectrogram, mel spectrogram, MFCC,
    pitch, energy, ZCR and spectral centroid on one shared time axis."""
    f = _feats(x, sr, feats)
    fig, ax = plt.subplots(8, 1, figsize=figsize, sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.4, 1.4, 1.4, 1, .8, .8, .8]})
    plot_waveform(x, sr, ax[0], title=title)
    plot_spectrogram(x, sr, ax[1])
    plot_mel_spectrogram(f, ax[2])
    plot_mfcc(f, ax[3])
    plot_pitch(f, ax[4], energy=False)
    plot_time_features(f, ax[5:])
    for a in ax:
        a.set_title(a.get_title(), fontsize=9, loc="left")
    for a in ax[:-1]:
        a.set_xlabel("")
    ax[5].set_title("")
    fig.tight_layout()
    return fig


def plot_wer_vs_snr(summary: Sequence[Dict[str, Any]], ax=None, *,
                    title: str = "Word error rate vs noise level", err_key: str = "wer_se"):
    """Line plot of WER (%) against SNR for each condition.

    ``summary`` rows need ``snr`` (a number, or ``"clean"``), ``condition`` (e.g.
    ``"raw"`` / ``"preprocessed"``) and ``wer`` (a fraction); an optional ``err_key`` column
    (default ``wer_se``, the standard error over clips) draws error bars.  "clean" is drawn
    at the left-hand end, as the best case.
    """
    ax = _ax(ax, figsize=(6.5, 4))
    snrs = sorted({r["snr"] for r in summary if r["snr"] != "clean"}, reverse=True)
    has_clean = any(r["snr"] == "clean" for r in summary)
    order = (["clean"] if has_clean else []) + snrs
    xpos = {s: i for i, s in enumerate(order)}
    style = {"raw": (COLORS["grey"], "o", "without preprocessing"),
             "preprocessed": (COLORS["blue"], "s", "with preprocessing")}
    for cond in dict.fromkeys(r["condition"] for r in summary):
        rows = sorted((r for r in summary if r["condition"] == cond), key=lambda r: xpos[r["snr"]])
        col, mk, lab = style.get(cond, (None, "^", cond))
        xs = [xpos[r["snr"]] for r in rows]
        ys = [100 * r["wer"] for r in rows]
        err = [100 * r.get(err_key, 0.0) for r in rows]
        ax.errorbar(xs, ys, yerr=err if any(err) else None, color=col, marker=mk, capsize=3,
                    label=lab)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["clean" if s == "clean" else f"{s:g} dB" for s in order])
    ax.set(title=title, xlabel="input SNR", ylabel="WER (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return ax
