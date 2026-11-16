"""
wer.py  --  SVARA, Member 2 / Sub-module 2C: Word Error Rate (the project's official WER)
=========================================================================================

Public contract (used by Member 1's 1C noise experiment, ``evaluate.py``)::

    from wer import compute_wer
    compute_wer("my name is athira", "my name athira mohan")      # -> 0.5

How WER is calculated
---------------------
The recogniser's output (the *hypothesis*) is lined up word by word against what was
really said (the *reference*), with the fewest possible edits.  Every difference is
one of three errors:

* **Substitution (S)**: a reference word was recognised as a different word
  ("twenty" -> "plenty").
* **Deletion (D)**: a reference word is missing from the hypothesis.
* **Insertion (I)**: the hypothesis has an extra word that was never said.

::

    WER = (S + D + I) / N        N = number of words in the REFERENCE

The fewest edits come from the **word-level Levenshtein (edit) distance**, worked out with
dynamic programming.  ``d[i][j]`` is the edit distance between the first ``i`` reference
words and the first ``j`` hypothesis words::

    d[i][0] = i                 (delete all i reference words)
    d[0][j] = j                 (insert all j hypothesis words)
    d[i][j] = min( d[i-1][j-1] + (0 if ref[i] == hyp[j] else 1),    match / substitution
                   d[i-1][j]   + 1,                                  deletion
                   d[i][j-1]   + 1 )                                 insertion

``d[N][M]`` is the total number of errors.  Tracing the path back from ``d[N][M]`` to
``d[0][0]`` splits that total into S, D and I (diagonal step = match or substitution,
step up = deletion, step left = insertion).  The path is not always unique, but every
minimal path has the same total, so the WER itself is always the same.

Notes
* WER can be **above 100 %** when the recogniser adds many words (insertions are not
  limited by N).
* An empty reference gives WER 0 if the hypothesis is empty too, otherwise 1.
* Both texts are **normalised** first (lower case, no punctuation; for English also
  "twenty five" == "25"), so formatting differences are not counted as errors.  Pass
  ``normalize=False`` to score the raw strings.
* :func:`compute_cer` is the same algorithm on characters (useful for Malayalam and
  Kannada, where one long word can carry a whole phrase).
* :func:`corpus_wer` adds up the errors of many sentences and divides by all reference
  words (the standard way to report a test set).

Command line::

    python wer.py "my name is athira" "my name athira mohan"            # alignment + matrix
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "WERResult", "normalize_text", "edit_distance_matrix", "align",
    "compute_wer", "compute_wer_full", "compute_cer", "corpus_wer",
]

# alignment operations
MATCH, SUB, DEL, INS = "match", "substitution", "deletion", "insertion"


# =========================================================================== normalisation
_EN_NORMALIZER: List[Any] = []


def _english_normalizer():
    """Whisper's EnglishTextNormalizer if installed (writes numbers as digits), else None."""
    if not _EN_NORMALIZER:
        try:
            from whisper.normalizers import EnglishTextNormalizer
            _EN_NORMALIZER.append(EnglishTextNormalizer())
        except ImportError:
            _EN_NORMALIZER.append(None)
    return _EN_NORMALIZER[0]


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
    s = unicodedata.normalize("NFC", s or "")
    english = language == "en" or (language is None and s.isascii())
    if english and _english_normalizer() is not None:
        s = _english_normalizer()(s)
        return re.sub(r"(?<=\d) (?=\d)", "", s).strip()
    s = s.casefold().replace("-", " ")
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] not in "PS")
    return re.sub(r"\s+", " ", s).strip()


# =========================================================================== the algorithm
def edit_distance_matrix(ref: Sequence[str], hyp: Sequence[str]) -> List[List[int]]:
    """The dynamic-programming table ``d`` (size (N+1) x (M+1)) described in the module doc."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost,      # match / substitution
                          d[i - 1][j] + 1,             # deletion
                          d[i][j - 1] + 1)             # insertion
    return d


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Trace back a minimal-edit path and return the alignment, left to right, as
    ``(operation, reference_word, hypothesis_word)``; the missing side is ``None``.

    Several paths can have the same (minimal) number of edits.  They all give the same
    WER, but to make the alignment easy to read this one keeps the most *correct* words:
    "is athira mohan" -> "athira mohanan kumar" is shown as D, match, S, I rather than
    S, S, S.  Remaining ties are read from the end of the sentence and prefer a deletion or
    insertion there, so "mohan kumar" vs "mohanan" pairs mohan with mohanan.
    """
    n, m = len(ref), len(hyp)
    # best[i][j] = (errors, -hits) of the best alignment of ref[:i] with hyp[:j]
    best = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0] = (i, 0)
    for j in range(1, m + 1):
        best[0][j] = (j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e, h = best[i - 1][j - 1]
            same = ref[i - 1] == hyp[j - 1]
            diag = (e + (not same), h - same)
            up = (best[i - 1][j][0] + 1, best[i - 1][j][1])
            left = (best[i][j - 1][0] + 1, best[i][j - 1][1])
            best[i][j] = min(diag, up, left)
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        diag_ok = same = False
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            e, h = best[i - 1][j - 1]
            diag_ok = best[i][j] == (e + (not same), h - same)
        if diag_ok and same:
            out.append((MATCH, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and best[i][j] == (best[i - 1][j][0] + 1, best[i - 1][j][1]):
            out.append((DEL, ref[i - 1], None))
            i -= 1
        elif j > 0 and best[i][j] == (best[i][j - 1][0] + 1, best[i][j - 1][1]):
            out.append((INS, None, hyp[j - 1]))
            j -= 1
        else:
            out.append((SUB, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
    out.reverse()
    return out


# =========================================================================== results
@dataclass
class WERResult:
    """Everything about one reference/hypothesis comparison."""
    wer: float                      # (S + D + I) / N
    substitutions: int              # S
    deletions: int                  # D
    insertions: int                 # I
    hits: int                       # words recognised correctly (H = N - S - D)
    n_ref: int                      # N, words in the reference
    n_hyp: int                      # M, words in the hypothesis
    reference: str                  # the reference after normalisation (what was scored)
    hypothesis: str                 # the hypothesis after normalisation
    alignment: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)
    unit: str = "word"              # "word" (WER) or "char" (CER)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def accuracy(self) -> float:
        """Word accuracy = 1 - WER (can be negative when there are many insertions)."""
        return 1.0 - self.wer

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["errors"] = self.errors
        d["alignment"] = [list(a) for a in self.alignment]
        return d

    def as_counts(self) -> Dict[str, Any]:
        """The short form used by evaluate.py: S, D, I, N, errors, wer."""
        return {"S": self.substitutions, "D": self.deletions, "I": self.insertions,
                "N": self.n_ref, "errors": self.errors, "wer": self.wer}

    def pretty(self) -> str:
        """Three aligned rows (REF / HYP / operation) for reports and the viva."""
        tag = {MATCH: "", SUB: "S", DEL: "D", INS: "I"}
        sep = "" if self.unit == "char" else " "
        cols = []
        for op, r, h in self.alignment:
            r, h = r if r is not None else "*" * len(h or ""), h if h is not None else "*" * len(r or "")
            w = max(len(r), len(h), 1)
            cols.append((r.ljust(w), h.ljust(w), tag[op].ljust(w)))
        rows = [sep.join(c[k] for c in cols).rstrip() for k in range(3)]
        name = "CER" if self.unit == "char" else "WER"
        return (f"REF: {rows[0]}\nHYP: {rows[1]}\n     {rows[2]}\n"
                f"S={self.substitutions} D={self.deletions} I={self.insertions} N={self.n_ref}  "
                f"{name} = ({self.substitutions}+{self.deletions}+{self.insertions})/{self.n_ref} = {self.wer:.2%}")


def _score(r: Sequence[str], h: Sequence[str], ref_text: str, hyp_text: str, unit: str) -> WERResult:
    path = align(r, h)
    S = sum(op == SUB for op, _, _ in path)
    D = sum(op == DEL for op, _, _ in path)
    I = sum(op == INS for op, _, _ in path)
    n = len(r)
    wer = (S + D + I) / n if n else float(len(h) > 0)
    return WERResult(wer=wer, substitutions=S, deletions=D, insertions=I, hits=n - S - D,
                     n_ref=n, n_hyp=len(h), reference=ref_text, hypothesis=hyp_text,
                     alignment=path, unit=unit)


# =========================================================================== public API
def compute_wer_full(reference: str, hypothesis: str, normalize: bool = True,
                     language: Optional[str] = None) -> WERResult:
    """WER with the full breakdown (S, D, I, N, alignment).  See the module doc."""
    if normalize:
        reference, hypothesis = normalize_text(reference, language), normalize_text(hypothesis, language)
    return _score(reference.split(), hypothesis.split(), reference, hypothesis, "word")


def compute_wer(reference: str, hypothesis: str, normalize: bool = True,
                language: Optional[str] = None) -> float:
    """The 2C contract: word error rate ``(S + D + I) / N`` as a fraction (0.25 = 25 %)."""
    return compute_wer_full(reference, hypothesis, normalize, language).wer


def compute_cer(reference: str, hypothesis: str, normalize: bool = True,
                language: Optional[str] = None) -> WERResult:
    """Character error rate: the same algorithm on characters (spaces are not counted).
    Characters are Unicode code points, so a Malayalam vowel sign counts as one character."""
    if normalize:
        reference, hypothesis = normalize_text(reference, language), normalize_text(hypothesis, language)
    r, h = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
    return _score(r, h, reference, hypothesis, "char")


Pair = Union[Tuple[str, str], WERResult, Dict[str, Any]]


def corpus_wer(items: Iterable[Pair], normalize: bool = True, language: Optional[str] = None) -> float:
    """Total errors / total reference words over a test set.  Items may be
    ``(reference, hypothesis)`` pairs, :class:`WERResult` objects, or dicts with
    ``errors`` and ``N`` (the rows evaluate.py writes).  Long sentences weigh more than
    short ones, unlike the mean of per-sentence WERs."""
    errors = words = 0
    for it in items:
        if isinstance(it, WERResult):
            e, n = it.errors, it.n_ref
        elif isinstance(it, dict):
            e, n = int(it["errors"]), int(it["N"])
        else:
            res = compute_wer_full(it[0], it[1], normalize, language)
            e, n = res.errors, res.n_ref
        errors += e
        words += n
    return errors / words if words else float("nan")


# =========================================================================== CLI
def _format_matrix(ref: Sequence[str], hyp: Sequence[str]) -> str:
    d = edit_distance_matrix(ref, hyp)
    w = max([len(x) for x in list(ref) + list(hyp)] + [3])
    head = " " * (w + 1) + " ".join(x.rjust(w) for x in ["#"] + list(hyp))
    lines = [head]
    for i, row in enumerate(d):
        label = "#" if i == 0 else ref[i - 1]
        lines.append(label.rjust(w) + " " + " ".join(str(v).rjust(w) for v in row))
    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - thin wrapper
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):          # Windows consoles: print Malayalam etc.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="SVARA 2C: word error rate between a reference and a hypothesis.")
    p.add_argument("reference")
    p.add_argument("hypothesis")
    p.add_argument("--language", help="en / hi / ml / kn (changes normalisation)")
    p.add_argument("--raw", action="store_true", help="do not normalise the texts")
    p.add_argument("--cer", action="store_true", help="also print the character error rate")
    a = p.parse_args(argv)
    r = compute_wer_full(a.reference, a.hypothesis, not a.raw, a.language)
    print(r.pretty())
    print("\nEdit-distance matrix d[i][j] (rows: reference, columns: hypothesis):")
    print(_format_matrix(r.reference.split(), r.hypothesis.split()))
    if a.cer:
        print("\n" + compute_cer(a.reference, a.hypothesis, not a.raw, a.language).pretty())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
