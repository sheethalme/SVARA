"""End-to-end check of 1A (audio_io) + 1B (preprocess) on ONE recording.

    python scripts/e2e_check.py                       # uses a built-in noisy test clip
    python scripts/e2e_check.py my_voice.wav          # any file: wav, mp3, flac, ogg (webm/m4a need ffmpeg)
    python scripts/e2e_check.py my_voice.wav --add-noise 10     # add 10 dB white noise to YOUR voice first
    python scripts/e2e_check.py my_voice.wav --out-dir results

It prints what each stage did, runs automatic PASS/FAIL checks on the output contract, and
writes into --out-dir (default: e2e_output/):
    cleaned.wav          the audio that would go to the speech recogniser  -> LISTEN TO IT
    input_16k.wav        the same recording after 1A only (for A/B listening)
    e2e_report.png       waveforms, spectrograms and the VAD decision
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import AudioLoadError, load_audio_with_report, save_wav, TARGET_SR
import preprocess as pp

OK, BAD = "PASS", "FAIL"


def add_white_noise(x, snr_db, seed=0):
    """Add white noise; SNR is measured against the power of the loud (speech) samples."""
    loud = x[np.abs(x) > 0.1 * np.max(np.abs(x))]
    p_sig = np.mean(loud.astype(np.float64) ** 2)
    noise = np.random.default_rng(seed).standard_normal(len(x))
    noise *= np.sqrt(p_sig / 10 ** (snr_db / 10)) / np.std(noise)
    return (x + noise).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="audio file (default: built-in noisy test clip)")
    ap.add_argument("--add-noise", type=float, metavar="SNR_DB", help="add white noise at this SNR first")
    ap.add_argument("--denoise", choices=["spectral_subtraction", "noisereduce", "none"], default="spectral_subtraction")
    ap.add_argument("--out-dir", default="e2e_output")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.input:
        src = a.input
        if not os.path.isfile(src):                       # never invent a file the user named
            print(f"{BAD}  file not found: {src}"); return 1
    else:
        src = os.path.join(ROOT, "sample_audio", "noisy", "white_10dB.wav")
        if not os.path.exists(src):                       # built-in demo clip: create if missing
            from make_noisy_samples import make_clip
            import soundfile as sf
            os.makedirs(os.path.dirname(src), exist_ok=True)
            sf.write(src, make_clip(10)[1] / 4, TARGET_SR)

    print(f"INPUT: {src}\n" + "=" * 70)
    # ---------------- 1A
    print("STEP 1A  audio_io.load_audio_with_report")
    try:
        x, sr, r1 = load_audio_with_report(src)
    except AudioLoadError as e:
        print(f"  {BAD}  could not load: {e}"); return 1
    print("  " + str(r1).replace("\n", "\n  "))
    if a.add_noise is not None:
        x = add_white_noise(x, a.add_noise)
        x = x / max(1e-9, np.max(np.abs(x))) * 0.9
        print(f"  (added white noise at {a.add_noise:g} dB SNR for the test)")
    save_wav(x, os.path.join(a.out_dir, "input_16k.wav"), sr)

    # ---------------- 1B
    print("\nSTEP 1B  preprocess.clean_audio_with_stages")
    res = pp.clean_audio_with_stages(x, sr, denoise=None if a.denoise == "none" else a.denoise)
    r2 = res.report
    print("  stages     :", " -> ".join(r2.stages_applied))
    print(f"  duration   : {r2.input_duration_s:.2f} s -> {r2.output_duration_s:.2f} s "
          f"(cut {r2.trimmed_start_s:.2f} s from the start, {r2.trimmed_end_s:.2f} s from the end)")
    print(f"  noise floor: {r2.noise_floor_before_db:.1f} dB -> {r2.noise_floor_after_db:.1f} dB "
          f"({r2.noise_floor_before_db - r2.noise_floor_after_db:+.1f} dB lower)")
    print(f"  speech     : detected={r2.speech_detected}, segments={r2.n_speech_segments}, VAD reliable={r2.vad_reliable}")
    for w in r2.warnings:
        print("  WARNING    :", w)
    cleaned = res.signal
    save_wav(cleaned, os.path.join(a.out_dir, "cleaned.wav"), res.sr)

    # ---------------- checks
    print("\nAUTOMATIC CHECKS (the contract Member 2 relies on)")
    checks = [
        ("output is a NumPy float32 array", isinstance(cleaned, np.ndarray) and cleaned.dtype == np.float32),
        ("mono (1-D)", cleaned.ndim == 1),
        ("sample rate is 16000 Hz", res.sr == 16000),
        ("all values finite", bool(np.all(np.isfinite(cleaned)))),
        ("values inside [-1, 1]", float(np.max(np.abs(cleaned))) <= 1.0),
        ("peak level is 0.9 (or clip is silent)", abs(float(np.max(np.abs(cleaned))) - 0.9) < 1e-3 or r2.warnings != [] and not r2.speech_detected),
        ("not empty (at least 0.1 s)", len(cleaned) >= 0.1 * res.sr),
        ("no longer than the input", len(cleaned) <= len(x)),
        ("stage signals are time-aligned", all(len(res.stages[k]) == len(x) for k in ("pre_emphasis", "bandpass", "denoised"))),
    ]
    if r2.speech_detected:
        checks.append(("speech detected and kept (output >= 0.3 s)", r2.output_duration_s >= 0.3))
    if r2.noise_reduction_applied:
        checks.append(("noise floor went down", r2.noise_floor_after_db < r2.noise_floor_before_db))
    failed = 0
    for name, ok in checks:
        print(f"  [{OK if ok else BAD}] {name}"); failed += (not ok)

    # ---------------- figure
    S = res.stages; t = lambda s: np.arange(len(s)) / sr
    fig, ax = plt.subplots(3, 2, figsize=(11, 8))
    ax[0, 0].plot(t(S["input"]), S["input"], lw=.3); ax[0, 0].set_title("BEFORE: after 1A (16 kHz mono)")
    off = res.trim_start / sr
    ax[0, 1].plot(np.arange(len(cleaned)) / sr + off, cleaned, lw=.3, color="C2"); ax[0, 1].set_title("AFTER: 1B cleaned (time axis kept)")
    ax[0, 1].set_xlim(ax[0, 0].get_xlim())
    for j, (s, o) in enumerate([(S["input"], 0.0), (cleaned, off)]):
        f, tt, P = sps.spectrogram(s, sr, nperseg=512, noverlap=384)
        ax[1, j].pcolormesh(tt + o, f, 10 * np.log10(P + 1e-12), vmin=-100, vmax=-20, shading="auto", cmap="magma")
        ax[1, j].set_ylabel("Hz"); ax[1, j].set_xlim(0, len(x) / sr)
    v = res.vad
    if v is not None:
        ax[2, 0].plot(v.frame_times_s, v.energy_db, lw=.9)
        ax[2, 0].axhline(v.threshold_db, color="r", ls="--", label="VAD threshold")
        ax[2, 0].fill_between(v.frame_times_s, v.energy_db.min(), v.energy_db.max(), where=v.mask, color="g", alpha=.2, label="speech")
        ax[2, 0].set_title("VAD decision (energy after noise reduction)"); ax[2, 0].legend(fontsize=7)
    f1, p1 = sps.welch(S["input"], sr, nperseg=1024); f2, p2 = sps.welch(cleaned, sr, nperseg=1024)
    ax[2, 1].plot(f1, 10 * np.log10(p1 + 1e-14), label="before"); ax[2, 1].plot(f2, 10 * np.log10(p2 + 1e-14), label="after")
    ax[2, 1].set_title("Average spectrum: 300-3400 Hz band kept"); ax[2, 1].set_xlabel("Hz"); ax[2, 1].legend(fontsize=7)
    fig.tight_layout(); png = os.path.join(a.out_dir, "e2e_report.png"); fig.savefig(png, dpi=110)

    print("\nFILES WRITTEN")
    for n in ("input_16k.wav", "cleaned.wav", "e2e_report.png"):
        print("  ", os.path.abspath(os.path.join(a.out_dir, n)))
    notes = []
    if not r2.speech_detected:
        notes.append("NO SPEECH WAS FOUND in this recording, so it was not trimmed. For a meaningful test, "
                     "record yourself with about half a second of silence before and after speaking.")
    if not r2.noise_reduction_applied and a.denoise != "none":
        notes.append("Noise reduction was skipped (it needs a short pause with only background sound).")
    if r1.warnings:
        notes.append("The input itself had problems (see WARNING above): clipping, too quiet, or silent.")
    print("\n" + ("CONTRACT CHECKS PASSED: output format is correct" if not failed
                  else f"CONTRACT CHECKS FAILED ({failed} check(s))"))
    for n in notes:
        print("NOTE:", n)
    if not failed and not notes:
        print("END-TO-END CHECK PASSED (speech found, noise reduced, silence trimmed)")
    print("Now LISTEN to input_16k.wav then cleaned.wav and look at e2e_report.png.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
