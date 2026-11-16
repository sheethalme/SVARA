"""Tests for 2C wer.py: the edit-distance WER used by the whole project."""
import random

import pytest

import wer as W


# ------------------------------------------------------------------ textbook cases
def test_identical_is_zero():
    assert W.compute_wer("my name is athira", "my name is athira") == 0.0


@pytest.mark.parametrize("ref,hyp,S,D,I", [
    ("my name is athira", "my name was athira", 1, 0, 0),            # substitution
    ("my name is athira", "my name athira", 0, 1, 0),                # deletion
    ("my name is athira", "my name is athira mohan", 0, 0, 1),       # insertion
    ("the cat sat on the mat", "the cat sit on mat today", 1, 1, 1),  # one of each
])
def test_counts_each_error_type(ref, hyp, S, D, I):
    r = W.compute_wer_full(ref, hyp, normalize=False)
    assert (r.substitutions, r.deletions, r.insertions) == (S, D, I)
    assert r.wer == pytest.approx((S + D + I) / len(ref.split()))
    assert r.hits == r.n_ref - S - D


def test_formula_uses_reference_length_and_can_exceed_one():
    assert W.compute_wer("yes", "yes yes yes") == 2.0                 # 2 insertions / 1 word
    assert W.compute_wer("a b", "c d e f", normalize=False) == 2.0    # 2 S + 2 I over N = 2


def test_empty_cases():
    assert W.compute_wer("", "") == 0.0
    assert W.compute_wer("", "hello") == 1.0
    assert W.compute_wer("hello world", "") == 1.0                    # all deletions


def test_wer_is_not_symmetric():
    assert W.compute_wer("a b c d", "a b", normalize=False) == 0.5    # 2 D / 4
    assert W.compute_wer("a b", "a b c d", normalize=False) == 1.0    # 2 I / 2


# ------------------------------------------------------------------ the DP table and the alignment
def test_edit_distance_matrix_borders_and_corner():
    ref, hyp = "kitten sat down".split(), "sitting sat".split()
    d = W.edit_distance_matrix(ref, hyp)
    assert [row[0] for row in d] == [0, 1, 2, 3]
    assert d[0] == [0, 1, 2]
    assert d[-1][-1] == 2                                              # S(kitten) + D(down)


def test_alignment_keeps_the_most_correct_words():
    r = W.compute_wer_full("my name is athira mohan", "my name athira mohanan kumar", normalize=False)
    assert r.errors == 3
    assert [op for op, _, _ in r.alignment] == ["match", "match", "deletion", "match", "insertion", "substitution"] \
        or [op for op, _, _ in r.alignment] == ["match", "match", "deletion", "match", "substitution", "insertion"]
    assert r.hits == 3


def test_alignment_reproduces_both_sentences():
    ref, hyp = "i am twenty five years old", "i am plenty five old years ago"
    r = W.compute_wer_full(ref, hyp, normalize=False)
    assert [a[1] for a in r.alignment if a[1] is not None] == ref.split()
    assert [a[2] for a in r.alignment if a[2] is not None] == hyp.split()


def test_pretty_shows_formula():
    out = W.compute_wer_full("my name is athira", "my name athira", normalize=False).pretty()
    assert "REF:" in out and "HYP:" in out and "D" in out
    assert "(0+1+0)/4 = 25.00%" in out


def _levenshtein(a, b):
    """Independent textbook implementation (two rows) to cross-check the totals."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def test_random_sentences_match_independent_levenshtein():
    rng = random.Random(7)
    vocab = "a b c d e f".split()
    for _ in range(300):
        ref = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
        r = W.compute_wer_full(" ".join(ref), " ".join(hyp), normalize=False)
        assert r.errors == _levenshtein(ref, hyp)
        assert r.hits == sum(op == "match" for op, _, _ in r.alignment)


def test_matches_jiwer_when_installed():
    jiwer = pytest.importorskip("jiwer")
    rng = random.Random(3)
    vocab = "my name is age phone twenty five old".split()
    for _ in range(100):
        ref = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 9)))
        hyp = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 9)))
        if not hyp:
            continue
        o = jiwer.process_words(ref, hyp)
        r = W.compute_wer_full(ref, hyp, normalize=False)
        assert r.wer == pytest.approx(o.wer)
        assert r.errors == o.substitutions + o.deletions + o.insertions


# ------------------------------------------------------------------ normalisation and languages
def test_normalisation_ignores_case_and_punctuation():
    assert W.compute_wer("My name is Athira.", "my name, is ATHIRA!") == 0.0
    assert W.compute_wer("My name is Athira.", "my name, is ATHIRA!", normalize=False) > 0


@pytest.mark.parametrize("ref,hyp,lang,expected", [
    ("मेरा नाम अथिरा है", "मेरा नाम अतिरा है।", "hi", 0.25),
    ("എന്റെ പേര് അഥിര", "എന്റെ പേര് അഥിര", "ml", 0.0),
    ("ನನ್ನ ಹೆಸರು ಅಥಿರಾ", "ನನ್ನ ಹೆಸರು", "kn", 1 / 3),
])
def test_indic_scripts(ref, hyp, lang, expected):
    assert W.compute_wer(ref, hyp, language=lang) == pytest.approx(expected)


def test_cer_counts_characters():
    r = W.compute_cer("athira", "adhira", normalize=False)
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 0)   # t -> d
    assert r.n_ref == 6 and r.unit == "char" and r.wer == pytest.approx(1 / 6)
    assert W.compute_cer("athira", "atira", normalize=False).deletions == 1
    assert W.compute_wer("athira", "adhira") == 1.0                     # the whole word is wrong
    assert W.compute_cer("a b", "ab").wer == 0.0                       # spaces are not counted


def test_corpus_wer_weights_by_words():
    pairs = [("a b c d e f g h i j", "a b c d e f g h i j"), ("yes", "no")]
    assert W.corpus_wer(pairs) == pytest.approx(1 / 11)
    results = [W.compute_wer_full(r, h) for r, h in pairs]
    assert W.corpus_wer(results) == pytest.approx(1 / 11)
    assert W.corpus_wer([r.as_counts() for r in results]) == pytest.approx(1 / 11)


def test_evaluate_uses_this_module():
    import evaluate
    assert evaluate.normalize_text is W.normalize_text
    assert evaluate.word_errors("my name is athira", "my name athira") == \
        {"S": 0, "D": 1, "I": 0, "N": 4, "errors": 1, "wer": 0.25}


def test_to_dict_is_serialisable():
    import json
    d = W.compute_wer_full("a b", "a c").to_dict()
    json.dumps(d)
    assert d["errors"] == 1 and d["alignment"][1] == ["substitution", "b", "c"]
