import csv, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_noisy_samples import make_clip  # noqa: E402

import evaluate as ev  # noqa: E402

SR = 16000
TTS = os.path.join(os.path.dirname(__file__), "..", "data", "test_clips_tts")


# ------------------------------------------------------------------ text + WER
def test_normalize_text_basic():
    # the basic (all-language) rules
    assert ev.normalize_text("  My Name, is ATHIRA-Mohan!! ", "hi") == "my name is athira mohan"
    assert ev.normalize_text("father's", "ml") == "fathers"
    # Indic scripts keep their vowel signs / viramas; danda (punctuation) is removed
    assert ev.normalize_text("मेरा नाम।") == "मेरा नाम"
    assert ev.normalize_text("എന്റെ പേര്") == "എന്റെ പേര്"


@pytest.mark.parametrize("ref,hyp", [
    ("i am twenty five years old", "I am 25 years old."),
    ("my date of birth is fifteenth march two thousand", "My date of birth is 15th March 2000."),
    ("i was born on the second of october nineteen eighty", "I was born on the 2nd of October 1980"),
    ("my phone number is nine eight four seven two one six five three zero",
     "My phone number is 98472 16530."),
    ("the pin code is five six zero zero two nine", "The PIN code is 560029."),
])
def test_english_numbers_are_normalised(ref, hyp):
    pytest.importorskip("whisper")
    assert ev.word_errors(ref, hyp, language="en")["wer"] == 0


def test_english_normaliser_not_applied_to_indic_text():
    assert ev.normalize_text("मेरी उम्र पच्चीस है") == "मेरी उम्र पच्चीस है"
    assert ev.normalize_text("ನನ್ನ ಹೆಸರು", "kn") == "ನನ್ನ ಹೆಸರು"


@pytest.mark.parametrize("ref,hyp,S,D,I", [
    ("a b c d", "a b c d", 0, 0, 0),
    ("a b c d", "a x c d", 1, 0, 0),
    ("a b c d", "a c d", 0, 1, 0),
    ("a b c d", "a b b c d", 0, 0, 1),
    ("a b c", "", 0, 3, 0),
    ("my name is ravi", "name is ravi kumar", 0, 1, 1),
])
def test_word_errors_counts(ref, hyp, S, D, I):
    e = ev.word_errors(ref, hyp, language="xx")
    assert (e["S"], e["D"], e["I"]) == (S, D, I)
    assert e["wer"] == pytest.approx((S + D + I) / len(ref.split()))


def test_wer_can_exceed_one_and_empty_reference():
    assert ev.word_errors("yes", "yes yes yes")["wer"] == 2.0
    assert ev.word_errors("", "")["wer"] == 0.0
    assert ev.word_errors("", "hello")["wer"] == 1.0


def test_word_errors_matches_jiwer_on_random_sentences():
    jiwer = pytest.importorskip("jiwer")
    rng = np.random.default_rng(0)
    vocab = "a b c d e f".split()
    for _ in range(300):
        r = " ".join(rng.choice(vocab, rng.integers(1, 9)))
        h = " ".join(rng.choice(vocab, rng.integers(0, 9)))
        o = jiwer.process_words(r, h)
        e = ev.word_errors(r, h, normalize=False)
        assert e["errors"] == o.substitutions + o.deletions + o.insertions
        assert e["wer"] == pytest.approx(o.wer)


def test_corpus_wer_weights_by_words():
    rows = [ev.word_errors("a b c d e f g h i j", "a b c d e f g h i j"),
            ev.word_errors("yes", "no")]
    assert ev.corpus_wer(rows) == pytest.approx(1 / 11)


# ------------------------------------------------------------------ noise mixing
@pytest.mark.parametrize("kind", ["white", "pink"])
@pytest.mark.parametrize("snr", [20, 10, 5, 0])
def test_add_noise_hits_requested_snr(kind, snr):
    clean, *_ = make_clip(100)
    noisy = ev.add_noise(clean, snr, kind, seed=3)
    assert noisy.dtype == np.float32 and noisy.shape == clean.shape
    # finite noise is never exactly uncorrelated with the speech: allow 0.2 dB
    assert ev.measured_snr(clean, noisy) == pytest.approx(snr, abs=0.2)
    assert np.max(np.abs(noisy)) <= 1.0


def test_snr_is_measured_on_speech_not_pauses():
    clean, *_ = make_clip(100, lead=3.0, trail=3.0)       # mostly silence
    noisy = ev.add_noise(clean, 10)
    noise = noisy - clean
    sp = slice(3 * SR, int(5.5 * SR))
    snr_during_speech = 10 * np.log10(np.mean(clean[sp].astype(float) ** 2) / np.mean(noise.astype(float) ** 2))
    assert snr_during_speech == pytest.approx(10, abs=0.5)


def test_pink_noise_spectrum_falls_3db_per_octave():
    n = ev.make_noise("pink", 2 ** 18, seed=1)
    P = np.abs(np.fft.rfft(n)) ** 2
    f = np.fft.rfftfreq(n.size, 1 / SR)
    band = lambda lo: P[(f >= lo) & (f < 2 * lo)].mean()
    assert 10 * np.log10(band(500) / band(1000)) == pytest.approx(3.0, abs=0.5)
    with pytest.raises(ValueError):
        ev.make_noise("brown", 10)


def test_noise_is_deterministic():
    clean, *_ = make_clip(100)
    np.testing.assert_array_equal(ev.add_noise(clean, 10, seed=5), ev.add_noise(clean, 10, seed=5))
    assert not np.array_equal(ev.add_noise(clean, 10, seed=5), ev.add_noise(clean, 10, seed=6))


# ------------------------------------------------------------------ experiment loop
def fake_asr(x, language=None):
    """Pretends to be a recogniser whose errors grow with the noise in the clip:
    it estimates the background level from the quietest frames."""
    from preprocess import frame_energy_db
    floor = np.percentile(frame_energy_db(x, 400, 160), 10)
    words = "my name is ravi kumar".split()
    n_wrong = int(np.clip((floor + 38) / 3, 0, len(words)))      # floor -38 dB: no errors
    return " ".join(["x"] * n_wrong + words[n_wrong:])


@pytest.fixture(scope="module")
def clips():
    out = []
    for i in range(3):
        clean, *_ = make_clip(100, seed=i)
        x = (clean / np.max(np.abs(clean)) * 0.9).astype(np.float32)
        out.append(ev.Clip(f"c{i}.wav", x, "my name is ravi kumar", "en"))
    return out


def test_run_experiment_rows_and_summary(clips):
    calls = []
    def asr(x, lang):
        calls.append(lang)
        return fake_asr(x)
    rows = ev.run_experiment(clips, asr, snrs=["clean", 20, 5], noise_kinds=["white", "pink"])
    # 3 clips x (clean once + 2 kinds x 2 SNRs) x 2 conditions
    assert len(rows) == 3 * (1 + 4) * 2 == len(calls)
    assert set(calls) == {"en"}
    assert {r["condition"] for r in rows} == {"raw", "preprocessed"}
    s = ev.summarize(rows)
    assert {(r["noise"], r["snr"]) for r in s} == {(k, x) for k in ("white", "pink")
                                                    for x in ("clean", 20.0, 5.0)}
    get = lambda k, snr, c: next(r for r in s if r["noise"] == k and r["snr"] == snr and r["condition"] == c)
    # more noise -> more errors without preprocessing; noise reduction lowers the floor -> fewer
    assert get("white", 5.0, "raw")["wer"] > get("white", 20.0, "raw")["wer"]
    assert get("white", 5.0, "preprocessed")["wer"] < get("white", 5.0, "raw")["wer"]
    assert get("white", "clean", "raw")["n"] == 3 and get("white", 5.0, "raw")["words"] == 15


def test_raw_and_preprocessed_get_the_same_noise(clips):
    seen = {}
    def asr(x, lang):
        seen.setdefault(len(seen) % 2, []).append(x)
        return ""
    ev.run_experiment(clips[:1], asr, snrs=[10])
    raw, pre = seen[0][0], seen[1][0]
    assert raw.shape[0] >= pre.shape[0]                 # VAD trimmed the preprocessed one
    assert np.max(np.abs(raw)) == pytest.approx(0.9, abs=1e-3)


def test_preprocess_overrides_are_applied_and_validated(clips):
    lens = []
    ev.run_experiment(clips[:1], lambda x, l: lens.append(len(x)) or "", snrs=["clean"],
                      preprocess={"vad": False})
    assert lens[0] == lens[1]                            # no trimming when VAD is off
    with pytest.raises(ValueError):
        ev.run_experiment(clips[:1], fake_asr, preprocess={"low_hz": -5})


def test_markdown_table(clips):
    s = ev.summarize(ev.run_experiment(clips[:1], fake_asr, snrs=["clean", 10]))
    md = ev.markdown_table(s)
    assert "| Clean |" in md and "| 10 dB |" in md and "WER (with preprocessing)" in md


def test_load_clips_and_csv_round_trip(tmp_path, clips):
    import soundfile as sf
    for c in clips:
        sf.write(tmp_path / c.name, c.signal, SR)
    with open(tmp_path / "references.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "transcript"])
        w.writerows([c.name, c.reference] for c in clips)
    got = ev.load_clips(str(tmp_path))
    assert [c.name for c in got] == [c.name for c in clips] and got[0].language is None
    assert len(ev.load_clips(str(tmp_path), limit=2)) == 2
    rows = ev.run_experiment(got[:1], fake_asr, snrs=["clean"])
    p = ev.write_csv(rows, str(tmp_path / "out" / "per_clip.csv"))
    assert len(list(csv.DictReader(open(p, encoding="utf-8")))) == 2
    png = ev.plot_summary(ev.summarize(rows), str(tmp_path / "out" / "wer.png"))
    assert os.path.getsize(png) > 5000


def test_load_clips_reports_missing_files(tmp_path):
    (tmp_path / "references.csv").write_text("file,transcript\nnope.wav,hello\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nope.wav"):
        ev.load_clips(str(tmp_path))
    (tmp_path / "bad.csv").write_text("name,text\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ev.load_clips(str(tmp_path), str(tmp_path / "bad.csv"))


def test_cli_value_parsing():
    assert ev._parse_value("none") is None and ev._parse_value("False") is False
    assert ev._parse_value("100") == 100 and ev._parse_value("0.5") == 0.5
    assert ev._parse_value("noisereduce") == "noisereduce"


# ------------------------------------------------------------------ real ASR (optional)
def _whisper_cached(name):
    return os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "whisper", f"{name}.pt"))


@pytest.mark.skipif(not (_whisper_cached("tiny.en") and os.path.isdir(TTS)),
                    reason="needs the cached Whisper tiny.en model and data/test_clips_tts")
def test_real_whisper_on_tts_clip():
    pytest.importorskip("whisper")
    transcribe, name = ev.get_transcriber("whisper", "tiny.en")
    clip = ev.load_clips(TTS, limit=4)[3]                # "i am twenty five years old"
    rows = ev.run_experiment([clip], transcribe, snrs=["clean"])
    assert name == "whisper-tiny.en"
    assert all(r["wer"] <= 0.25 for r in rows), [r["hypothesis"] for r in rows]
