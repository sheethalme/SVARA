"""
normalizers.py  --  SVARA 2B: turn *spoken* numbers, digits and dates into machine values
========================================================================================

Pure functions, no network, no model.  They work on the text a speech recogniser produces
("my age is twenty five", "पच्चीस साल", "five march nineteen ninety", "double nine eight ...").

Main pieces
-----------
* :func:`tokenize`         -- split into word tokens (keeps Indic combining marks and zero-width joiners).
* :func:`number_spans`     -- find numbers, written as digits ("25", "1,50,000", "5th") or as words
                              ("twenty five", "two thousand and five", "पच्चीस", "एक लाख पचास हज़ार").
* :func:`parse_number`     -- first number in a range (ages, amounts).
* :func:`spoken_digits`    -- digit-by-digit speech for phone / PIN / Aadhaar ("double nine", "डबल पाँच").
* :func:`parse_date`       -- day-first dates: "5/3/1990", "5th March 1990", "five march nineteen ninety",
                              "५ मार्च १९९०", "tomorrow".
* :func:`verhoeff_valid`   -- the checksum used by Aadhaar numbers (catches most mis-heard digits).
"""

from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from lexicon import (DOUBLE_WORDS, DIGIT_EXTRA, MONTH_PREFIXES, MONTH_WORDS, MULTIPLIERS, NUMBER_WORDS,
                     RELATIVE_DAYS, ascii_digits, fold)

__all__ = ["tokenize", "NumSpan", "number_spans", "parse_number", "spoken_digits", "DateParse", "parse_date",
           "verhoeff_valid", "verhoeff_check_digit", "prep"]


# =========================================================================== tokens
def _is_tok_char(c: str) -> bool:
    # letters, combining marks (matras, viramas), numbers, apostrophes, zero-width (joiners inside Indic words)
    return unicodedata.category(c)[0] in "LMN" or c in "'’‌‍"


def prep(text: str) -> str:
    """ASCII digits and no thousands separators ("1,50,000" -> "150000"); everything else untouched."""
    text = unicodedata.normalize("NFC", text).replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"(?<=\d),(?=\d)", "", ascii_digits(text))


def tokenize(text: str) -> List[str]:
    """Split on anything that cannot be part of a word.  Original spelling is kept (use :func:`fold`
    on the tokens to compare them with the word lists)."""
    toks, cur = [], []
    for c in prep(text):
        if _is_tok_char(c):
            cur.append(c)
        elif cur:
            toks.append("".join(cur)); cur = []
    if cur:
        toks.append("".join(cur))
    return toks


# =========================================================================== numbers
@dataclass
class NumSpan:
    first: int                 # index of the first token of the number
    last: int                  # index of its last token
    value: int
    ordinal: bool = False      # "5th", "fifth"
    words: bool = False        # written in words (False: digits)
    raw: str = ""              # digit string as written (keeps leading zeros)


_ORD_DIGITS = re.compile(r"^(\d+)(st|nd|rd|th)$")


def number_spans(ftokens: List[str]) -> List[NumSpan]:
    """Find every number in a list of *folded* tokens.

    Grammar for words: a number may continue only when
      * the previous word was a multiplier ("hundred", "सौ", "lakh", ...)  -> "two hundred five"
      * the next word is a multiplier                                       -> "five hundred", "उन्नीस सौ"
      * the previous word was a multiple of ten and the next is 1-9         -> "twenty five"
    so "nine eight seven" is three numbers (digit-by-digit speech), "nineteen ninety" is two (a year said
    in two halves), and "पच्चीस" is one.  The connector "and" is allowed after a multiplier.
    """
    spans: List[NumSpan] = []
    n, i = len(ftokens), 0
    while i < n:
        t = ftokens[i]
        if t.isdigit():
            spans.append(NumSpan(i, i, int(t), False, False, t)); i += 1; continue
        m = _ORD_DIGITS.match(t)
        if m:
            spans.append(NumSpan(i, i, int(m.group(1)), True, False, m.group(1))); i += 1; continue
        starts_a = t == "a" and i + 1 < n and NUMBER_WORDS.get(ftokens[i + 1], (0, "", False))[1] == "mult"
        if t not in NUMBER_WORDS and not starts_a:
            i += 1; continue
        j, total, cur, prev, ordinal = i, 0, 0, None, False
        while j < n:
            tok = ftokens[j]
            if tok == "and" and prev is not None and prev[1] == "mult" and j + 1 < n \
                    and NUMBER_WORDS.get(ftokens[j + 1], (0, "mult", False))[1] != "mult":
                j += 1; continue
            if tok == "a" and prev is None and j == i and starts_a:
                v, k, o = 1, "unit", False
            elif tok in NUMBER_WORDS:
                v, k, o = NUMBER_WORDS[tok]
            else:
                break
            if prev is not None:
                pv, pk = prev
                if not (k == "mult" or pk == "mult" or (pk == "tens" and k == "unit" and v >= 1)):
                    break
            if k == "mult":
                if v == 100:
                    cur = (cur or 1) * 100
                else:
                    total += (cur or 1) * v; cur = 0
            else:
                cur += v
            ordinal = ordinal or o
            prev = (v, k); j += 1
        if j == i:                      # lone "and" / defensive: always make progress
            i += 1; continue
        # trailing "and" was consumed only when followed by a number, so j-1 is the last number word
        spans.append(NumSpan(i, j - 1, total + cur, ordinal, True, ""))
        i = j
    return spans


def parse_number(text: str, lo: Optional[int] = None, hi: Optional[int] = None) -> Optional[int]:
    """First number in ``text`` (digits or words) whose value lies in ``[lo, hi]``."""
    ft = [fold(t) for t in tokenize(text)]
    for s in number_spans(ft):
        if (lo is None or s.value >= lo) and (hi is None or s.value <= hi):
            return s.value
    return None


def spoken_digits(text: str) -> str:
    """The digits of a phone / PIN / Aadhaar number read out loud.

    Handles ASCII and Indic digits ("९८७६"), digit words in four languages, ``double`` / ``triple``
    ("double nine" -> 99, "डबल पाँच" -> 55), English "oh" for zero, and two-digit words ("ninety eight" -> 98).
    Everything that is not a number (filler words, "my number is", "plus") is ignored."""
    ft = [fold(t) for t in tokenize(text)]
    spans = {s.first: s for s in number_spans(ft)}
    out: List[str] = []
    i, n = 0, len(ft)

    def digit_at(k: int) -> Optional[str]:
        if k in spans and (spans[k].raw or 0 <= spans[k].value <= 9):
            s = spans[k]
            return s.raw if s.raw else str(s.value)
        if k < n and ft[k] in DIGIT_EXTRA:
            return str(DIGIT_EXTRA[ft[k]])
        return None

    while i < n:
        t = ft[i]
        if t in DOUBLE_WORDS and i + 1 < n:
            d = digit_at(i + 1)
            if d is not None and len(d) == 1:
                out.append(d * DOUBLE_WORDS[t]); i += 2; continue
        if i in spans:
            s = spans[i]
            out.append(s.raw if s.raw else str(s.value)); i = s.last + 1; continue
        if t in DIGIT_EXTRA:               # "oh" / "o": zero only when it sits next to other digits
            prev_is_digit = bool(out) and (i - 1 >= 0) and ((i - 1) in {s.last for s in spans.values()} or ft[i - 1] in DIGIT_EXTRA)
            next_is_digit = (i + 1) in spans or (i + 1 < n and (ft[i + 1] in DIGIT_EXTRA or ft[i + 1] in DOUBLE_WORDS))
            if prev_is_digit or next_is_digit:
                out.append(str(DIGIT_EXTRA[t]))
        i += 1
    return "".join(out)


# =========================================================================== dates
@dataclass
class DateParse:
    date: Optional[_dt.date] = None
    ambiguous: bool = False            # day/month order or a two-digit year had to be guessed
    issues: List[str] = field(default_factory=list)


_ISO = re.compile(r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)")
_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})(?!\d)")


def _expand_year(yy: int, today: _dt.date) -> int:
    return 2000 + yy if 2000 + yy <= today.year else 1900 + yy


def _make(y: int, m: int, d: int, res: DateParse) -> DateParse:
    try:
        res.date = _dt.date(y, m, d)
    except ValueError:
        res.issues.append(f"{d}/{m}/{y} is not a real date")
    return res


def _month_at(tok: str) -> Optional[int]:
    if tok in MONTH_WORDS:
        return MONTH_WORDS[tok]
    for stem, m in MONTH_PREFIXES:
        if tok.startswith(stem):
            return m
    return None


def parse_date(text: str, today: Optional[_dt.date] = None) -> DateParse:
    """Parse a spoken or written date (day-first, as in India).  Returns a :class:`DateParse`;
    ``date`` is None (with ``issues``) when no complete date could be read.  ``ambiguous`` is set when the
    day/month order or a two-digit year was a guess (always read such a date back to the user)."""
    today = today or _dt.date.today()
    res = DateParse()
    t = fold(prep(text))

    for word, off in RELATIVE_DAYS.items():                       # English only
        if re.search(rf"(?<![a-z]){word}(?![a-z])", t):
            res.date = today + _dt.timedelta(days=off)
            return res

    m = _ISO.search(t)
    if m:
        return _make(int(m.group(1)), int(m.group(2)), int(m.group(3)), res)
    m = _DMY.search(t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if mo > 12 >= d:                                           # 03/25/1990: month-first
            d, mo = mo, d; res.ambiguous = True
        elif d <= 12 and mo <= 12 and d != mo:
            res.ambiguous = True                                   # 05/03/1990 could be 3 May or 5 March
        yy = int(y)
        if len(y) == 2:
            yy = _expand_year(yy, today); res.ambiguous = True
        return _make(yy, mo, d, res)

    ft = tokenize(t)
    spans = number_spans(ft)
    month_idx, month = None, None
    for i, tok in enumerate(ft):
        mm = _month_at(tok)
        if mm:
            month_idx, month = i, mm; break

    used: set = set()
    year: Optional[int] = None
    for k, s in enumerate(spans):                                  # 1) a whole year: 1990, "two thousand five"
        if 1800 <= s.value <= 2100 and not s.ordinal:
            year = s.value; used.add(k); break
    if year is None:                                               # 2) a year said in two halves
        for k in range(len(spans) - 1):
            a, b = spans[k], spans[k + 1]
            if a.value in (19, 20) and not a.ordinal:
                if b.first == a.last + 1 and 0 <= b.value <= 99 and not b.ordinal:
                    year = a.value * 100 + b.value; used |= {k, k + 1}; break
                if b.first == a.last + 2 and ft[a.last + 1] in DIGIT_EXTRA and 1 <= b.value <= 9:
                    year = a.value * 100 + b.value; used |= {k, k + 1}; break          # "twenty oh five"
        else:
            for k in range(len(spans) - 2):                        # "twenty zero five"
                a, z, b = spans[k], spans[k + 1], spans[k + 2]
                if a.value in (19, 20) and z.value == 0 and z.words and b.value and 1 <= b.value <= 9 \
                        and z.first == a.last + 1 and b.first == z.last + 1:
                    year = a.value * 100 + b.value; used |= {k, k + 1, k + 2}; break

    if month is not None:
        # the day is the 1-31 number nearest to the month name (before it is preferred)
        cands = [(k, s) for k, s in enumerate(spans) if k not in used and 1 <= s.value <= 31]
        day = None
        if cands:
            def dist(item):
                _, s = item
                before = month_idx - s.last
                after = s.first - month_idx
                return (0, before) if before > 0 else (1, after)
            k, s = min(cands, key=dist)
            day = s.value; used.add(k)
        if year is None:                                           # 3) a two-digit year after the month ("5 march 90")
            late = [(k, s) for k, s in enumerate(spans) if k not in used and s.first > month_idx and 0 <= s.value <= 99
                    and not s.ordinal]
            if late and day is not None:
                year = _expand_year(late[-1][1].value, today); res.ambiguous = True
        if day is None:
            res.issues.append("I could not hear the day of the month")
        if year is None:
            res.issues.append("I could not hear the year")
        if day is not None and year is not None:
            return _make(year, month, day, res)
        return res

    # no month name: three numbers "5 3 1990" -> day, month, year
    nums = [s for k, s in enumerate(spans) if k not in used]
    if year is not None and len(nums) >= 2:
        d, mo = nums[0].value, nums[1].value
        res.ambiguous = True
        return _make(year, mo, d, res)
    res.issues.append("I could not find a date")
    return res


# =========================================================================== Verhoeff (Aadhaar checksum)
_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    """True if the digit string passes the Verhoeff checksum (Aadhaar numbers: 12 digits incl. check digit)."""
    if not number.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> int:
    """Check digit to append to ``number`` (used by the tests to build valid Aadhaar-like numbers)."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]
