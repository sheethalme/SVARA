import os, sys
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_noisy_samples import make_clip  # noqa: E402

import plots  # noqa: E402
from features import extract_features  # noqa: E402
from preprocess import clean_audio_with_stages  # noqa: E402

SR = 16000


@pytest.fixture(scope="module")
def result():
    _, noisy, _, _ = make_clip(10)
    return clean_audio_with_stages(noisy, SR)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_before_after_layout_and_shared_axes(result):
    fig = plots.plot_before_after(result)
    axes = fig.axes
    main = [a for a in axes if a.get_ylabel() != "energy (dB)"]       # minus pitch twin axes
    assert len(main) == 8
    titles = [a.get_title() for a in main]
    assert titles[:2] == ["Before preprocessing", "After preprocessing"]
    assert {"Spectrogram", "MFCC (c1-c12)", "Pitch contour (F0)"} <= set(titles)
    xl = {tuple(np.round(a.get_xlim(), 6)) for a in main}
    assert len(xl) == 1 and xl.pop() == pytest.approx((0, len(result.stages["input"]) / SR))
    # same colour scale for both spectrograms and both MFCC maps
    by_title = {}
    for a in main:
        if a.collections:
            by_title.setdefault(a.get_title(), []).append(a.collections[0].get_clim())
    assert by_title["Spectrogram"][0] == pytest.approx(by_title["Spectrogram"][1])
    # MFCC maps: own symmetric scale per column (baselines differ after pre-emphasis/band-pass)
    for lo, hi in by_title["MFCC (c1-c12)"]:
        assert lo == pytest.approx(-hi)


def test_after_column_is_placed_at_trim_offset(result):
    fig = plots.plot_before_after(result)
    after_wave = fig.axes[1].lines[0]
    t = after_wave.get_xdata()
    assert t[0] == pytest.approx(result.trim_start / SR)
    assert t[-1] == pytest.approx(result.trim_end / SR - 1 / SR, abs=1e-3)


def test_before_after_with_arrays():
    x = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
    fig = plots.plot_before_after(x, x[4000:12000], offset_s=0.25)
    assert fig.axes[1].lines[0].get_xdata()[0] == pytest.approx(0.25)


def test_single_panels_return_axes():
    _, noisy, _, _ = make_clip(20)
    f = extract_features(noisy)
    for ax in (plots.plot_waveform(noisy), plots.plot_spectrogram(noisy, colorbar=True),
               plots.plot_mel_spectrogram(f, colorbar=True), plots.plot_mfcc(f, colorbar=True),
               plots.plot_pitch(f)):
        assert isinstance(ax, plt.Axes)
    axes = plots.plot_time_features(f)
    assert [a.get_ylabel() for a in axes] == ["energy (dB)", "ZCR", "centroid (Hz)"]


def test_mfcc_plot_hides_c0_by_default():
    f = extract_features(make_clip(20)[1])
    assert plots.plot_mfcc(f).collections[0].get_array().size == 12 * f["times"].size
    assert plots.plot_mfcc(f, skip_c0=False).collections[0].get_array().size == 13 * f["times"].size


def test_pitch_plot_draws_only_voiced():
    f = extract_features(make_clip(20)[1])
    y = plots.plot_pitch(f, energy=False).lines[0].get_ydata()
    assert np.isnan(y).sum() == (~f["voiced"]).sum()


def test_vad_plot(result):
    ax = plots.plot_vad(result.vad, trim=(result.trim_start, result.trim_end))
    assert ax.get_legend() is not None


def test_feature_overview(tmp_path):
    fig = plots.plot_feature_overview(make_clip(20)[1])
    p = tmp_path / "overview.png"
    fig.savefig(p)
    assert p.stat().st_size > 10_000


def test_wer_vs_snr_plot():
    summary = [{"snr": s, "condition": c, "wer": w, "wer_se": 0.05}
               for s, c, w in [("clean", "raw", .05), ("clean", "preprocessed", .06),
                               (20, "raw", .1), (20, "preprocessed", .08),
                               (5, "raw", .6), (5, "preprocessed", .4)]]
    ax = plots.plot_wer_vs_snr(summary)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["clean", "20 dB", "5 dB"]
    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["without preprocessing", "with preprocessing"]
    ys = np.asarray(ax.containers[0].lines[0].get_ydata(), float)     # the 'raw' data line
    np.testing.assert_allclose(ys, [5, 10, 60])
