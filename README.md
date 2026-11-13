# SVARA — Parts 1B and 1C: Speech Preprocessing, Features and Evaluation

## What Part 1B does

Part 1B prepares audio for a speech-recognition system. It takes audio accepted by
Part 1A (`audio_io.py`) and returns a cleaned, mono, 16 kHz `float32` signal with
its peak normalised to 0.9. The main entry point is:

```python
from preprocess import clean_audio

cleaned = clean_audio("answer.wav")
```

The preprocessing chain is:

```text
Part 1A audio → pre-emphasis → band-pass filter → noise reduction
              → voice activity detection (VAD) → re-normalisation
```

## Work completed

1. **Pre-emphasis:** Applies `y[n] = x[n] - 0.97 * x[n-1]` to reduce low-frequency
   dominance and bring out higher-frequency speech cues.
2. **Band-pass filtering:** Uses a fourth-order Butterworth filter from 300 to
   3400 Hz. It is applied forward and backward, so it does not shift speech in
   time.
3. **Noise reduction:** The default method is an implemented spectral-subtraction
   algorithm. It estimates noise from quieter frames, subtracts it from the
   signal spectrum, and smooths the gain to reduce musical noise. An optional
   `noisereduce` backend is also available.
4. **Speech detection and trimming:** An energy-based VAD finds speech using an
   adaptive threshold. It trims leading and trailing silence while keeping a
   150 ms margin and preserving pauses inside an utterance. When speech cannot
   be distinguished reliably from the background, it leaves the clip untrimmed.
5. **Output and reporting:** The result is re-normalised and kept in the range
   `[-1, 1]`. A detailed mode returns intermediate signals, VAD decisions, and a
   report with applied stages, durations, noise-floor measurements, and warnings.

Each processing stage can be disabled or tuned with `PreprocessConfig` (or
keyword overrides), which supports comparing the full chain with individual
stages:

```python
from preprocess import clean_audio_with_stages

result = clean_audio_with_stages("answer.wav", denoise=None, vad=False)
cleaned = result.signal
before_denoising = result.stages["bandpass"]
```

## Evaluation and tests

The pipeline was tested using synthetic speech-like clips mixed with white or
pink noise at 20, 10, and 5 dB SNR. Across those tests, VAD found speech in all
five seeds per noise/SNR combination, with mean boundary errors of 12–18 ms.
Spectral subtraction improved SI-SNR over filtering alone by approximately
6–8 dB at 10 and 5 dB input SNR. See [`docs/results_1b.csv`](docs/results_1b.csv)
for the measurements and [`docs/1B_Preprocessing_Chain.md`](docs/1B_Preprocessing_Chain.md)
for the detailed design, figures, and test descriptions.

The test suite includes checks for filter responses, noise reduction, VAD
boundaries and fallback behavior, supported input types, output format, and
configuration validation. Run it with:

```powershell
python -m pytest
```

The repository currently collects 215 tests across Parts 1A, 1B and 1C.

**Evaluation limitation:** these performance measurements use synthetic
speech-like signals and synthetic noise, not real recorded speech. The stages
and their parameters should be evaluated on real recordings before drawing
conclusions about recognition accuracy.

## Try it

Install the project dependencies and run the test suite:

```powershell
pip install -r requirements.txt
python -m pytest
```

To preprocess a file and save the result:

```powershell
python preprocess.py input.wav cleaned.wav
```

To run the end-to-end check and generate example audio and a diagnostic plot:

```powershell
python scripts/e2e_check.py
```

## Main Part 1B files

- `preprocess.py` — preprocessing stages, configuration, reports, and command-line interface.
- `tests/test_preprocess.py` — Part 1B tests.
- `scripts/demo_preprocess.py` — generates evaluation figures and the results CSV.
- `scripts/make_noisy_samples.py` — creates synthetic noisy test clips.
- `docs/1B_Preprocessing_Chain.md` — detailed implementation and evaluation notes.
- `docs/results_1b.csv` and `docs/figures/` — measured results and figures.

---

# Part 1C: Feature Extraction, Visualisation and Evaluation

## What Part 1C does

```python
from preprocess import clean_audio_with_stages
from features import extract_features
from plots import plot_before_after

feats = extract_features(cleaned)           # dict of arrays on one 25 ms / 10 ms frame grid
fig = plot_before_after(clean_audio_with_stages("answer.wav"))   # Signal Lab figure
```

1. **Features (`features.py`):** short-time energy, zero-crossing rate, spectral centroid,
   pitch F0 (YIN), 40-band mel spectrogram, 13 MFCCs with delta and delta-delta. Every step
   (framing, Hamming window, mel filterbank, log, DCT, YIN) is implemented from its formula and
   checked against librosa in the tests.
2. **Plots (`plots.py`):** before/after waveform, spectrogram, MFCC heat-map and pitch contour
   on a shared time axis; a full feature overview; the WER-vs-SNR chart.
3. **Evaluation (`evaluate.py`):** adds white or pink noise at 20/10/5 dB SNR (measured on the
   speech frames), transcribes each clip with and without 1B, and reports WER. It uses Member 2's
   `asr_agent.transcribe` when it exists, otherwise local Whisper. `--set field=value` lets 1B
   tune the chain.

## Main result so far (stand-in clips)

On 24 TTS stand-in clips with Whisper `base.en`, **the 1B chain increases WER** (5 dB white
noise: 18.4 % → 31.3 %). The ablation shows pre-emphasis and the narrow 300–3400 Hz band are the
main causes, while spectral subtraction helps within the chain. Recommendation: feed Whisper the
1A output (optionally VAD-trimmed) and keep 1B for the Signal Lab and features. This must be
re-checked on 1A's real recordings. Details, figures and viva notes are in
[`docs/1C_Feature_Extraction_and_Evaluation.md`](docs/1C_Feature_Extraction_and_Evaluation.md).

## Try it

```powershell
python scripts/e2e_check.py my_voice.wav      # now also writes signal_lab.png
streamlit run scripts/try_it_app.py           # now has a 1C Signal Lab section
python scripts/make_tts_clips.py              # stand-in clips (Windows) until real ones exist
python evaluate.py --clips data/test_clips_tts --model base.en --noise white pink
python evaluate.py --clips data/test_clips --model small    # real clips + references.csv
```

The real experiment needs 1A's 20–30 recordings in `data/test_clips/` with a `references.csv`
(`file,transcript,language`).

## Main Part 1C files

- `features.py`, `plots.py`, `evaluate.py` — the three 1C modules.
- `tests/test_features.py`, `tests/test_plots.py`, `tests/test_evaluate.py` — 93 tests.
- `scripts/demo_features.py` — figures 9–12 and `docs/results_1c_features.csv`.
- `scripts/make_tts_clips.py`, `data/test_clips_tts/` — stand-in evaluation clips.
- `docs/results_1c_wer.csv`, `docs/results_1c_ablation.csv`, `docs/figures/fig9`–`fig13`.
