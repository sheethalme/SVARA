"""
lexicon.py  --  SVARA 2B: word lists (data only, no logic) for English, Hindi, Malayalam, Kannada
=================================================================================================

Everything here is *matched after* :func:`fold`, which makes spelling variants equal:
Unicode NFC, zero-width joiners removed, old-style Malayalam chillu letters turned into the modern
atomic letters, the Devanagari nukta dropped (``ज़`` -> ``ज``), chandrabindu -> anusvara
(``पाँच`` -> ``पांच``), digits of any script -> ASCII, lower case.

Coverage and honesty:
* English and Hindi tables are complete for the cases a form needs (numbers 0-99 plus hundred /
  thousand / lakh / crore; months; gender; yes / no / "say again").
* Malayalam and Kannada tables cover the base number words, the tens, the regular compound forms
  (``ഇരുപത്തിയഞ്ച്``, ``ಇಪ್ಪತ್ತೈದು``), months, gender and confirmations.  They were written from
  language references and NOT reviewed by native speakers or tested on real recogniser output; spoken
  Malayalam / Kannada numbers also have many dialect forms.  Whisper usually writes numbers as digits,
  which are handled in every script.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- folding
_ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)
_CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ", "ല്‍": "ൽ", "ള്‍": "ൾ", "ക്‍": "ൿ"}


def ascii_digits(s: str) -> str:
    """Digits of any script (Devanagari ०-९, Malayalam ൦-൯, Kannada ೦-೯, ...) -> ASCII 0-9."""
    return "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" else c for c in s)


def fold(s: str) -> str:
    """Normalise text for *matching* (never for display)."""
    s = unicodedata.normalize("NFC", s)
    for old, new in _CHILLU.items():
        s = s.replace(old, new)
    s = s.translate(_ZW).replace("़", "").replace("ँ", "ं")
    return ascii_digits(s).casefold()


def _f(words) -> List[str]:
    return [fold(w) for w in words]


# --------------------------------------------------------------------------- number words
# folded word -> (value, kind, ordinal)
#   kind: "unit" 0-9 | "whole" a self-contained 10-99 (cannot take a following unit) |
#         "tens" 20,30..90 that MAY be followed by a unit ("twenty five") | "mult" 100, 1000, ...
NUMBER_WORDS: Dict[str, Tuple[int, str, bool]] = {}


def _add(table: Dict[str, int], kind_for, ordinal: bool = False) -> None:
    for w, v in table.items():
        NUMBER_WORDS[fold(w)] = (v, kind_for(v), ordinal)


_en_kind = lambda v: "unit" if v < 10 else "whole" if v < 20 else "tens" if v % 10 == 0 else "whole"
_hi_kind = lambda v: "unit" if v < 10 else "whole"
_ind_kind = lambda v: "unit" if v < 10 else "whole" if v < 20 else "tens" if v % 10 == 0 else "whole"

EN_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
          "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
          "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
          "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
          "twentieth": 20, "thirtieth": 30}

HI_NUM = {
    "शून्य": 0, "ज़ीरो": 0, "जीरो": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6,
    "छे": 6, "छ": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "पंदरह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28,
    "अठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "इकत्तीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37,
    "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चौवालीस": 44, "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46,
    "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "तिरेपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56, "सत्तावन": 57,
    "अट्ठावन": 58, "अठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68,
    "उनहत्तर": 69, "सत्तर": 70,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77,
    "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88,
    "अठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97,
    "अट्ठानवे": 98, "अठानवे": 98, "निन्यानवे": 99, "निन्यानबे": 99,
}

ML_UNITS = {"പൂജ്യം": 0, "ഒന്ന്": 1, "രണ്ട്": 2, "മൂന്ന്": 3, "നാല്": 4, "അഞ്ച്": 5, "ആറ്": 6, "ഏഴ്": 7, "എട്ട്": 8,
            "ഒമ്പത്": 9, "ഒൻപത്": 9}
ML_TEENS = {"പത്ത്": 10, "പതിനൊന്ന്": 11, "പന്ത്രണ്ട്": 12, "പതിമൂന്ന്": 13, "പതിനാല്": 14, "പതിനഞ്ച്": 15,
            "പതിനാറ്": 16, "പതിനേഴ്": 17, "പതിനെട്ട്": 18, "പത്തൊമ്പത്": 19, "പത്തൊൻപത്": 19}
ML_TENS = {"ഇരുപത്": 20, "മുപ്പത്": 30, "നാൽപ്പത്": 40, "നാല്പത്": 40, "അമ്പത്": 50, "അറുപത്": 60, "എഴുപത്": 70,
           "എൺപത്": 80, "തൊണ്ണൂറ്": 90}
ML_MULT = {"നൂറ്": 100, "നൂറു": 100, "ആയിരം": 1000, "ലക്ഷം": 100000, "കോടി": 10 ** 7}
# regular compounds: tens-stem + (യ + unit when the unit starts with a vowel) -> ഇരുപത്തിയഞ്ച് = 25
ML_COMPOUND_STEMS = {"ഇരുപത്തി": 20, "മുപ്പത്തി": 30, "നാൽപ്പത്തി": 40, "നാല്പത്തി": 40, "അമ്പത്തി": 50, "അറുപത്തി": 60,
                     "എഴുപത്തി": 70, "എൺപത്തി": 80, "തൊണ്ണൂറ്റി": 90}
ML_COMPOUND_UNITS = {1: "യൊന്ന്", 2: "രണ്ട്", 3: "മൂന്ന്", 4: "നാല്", 5: "യഞ്ച്", 6: "യാറ്", 7: "യേഴ്", 8: "യെട്ട്",
                     9: "യൊമ്പത്"}

KN_UNITS = {"ಸೊನ್ನೆ": 0, "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲ್ಕು": 4, "ಐದು": 5, "ಆರು": 6, "ಏಳು": 7, "ಎಂಟು": 8,
            "ಒಂಬತ್ತು": 9}
KN_TEENS = {"ಹತ್ತು": 10, "ಹನ್ನೊಂದು": 11, "ಹನ್ನೆರಡು": 12, "ಹದಿಮೂರು": 13, "ಹದಿನಾಲ್ಕು": 14, "ಹದಿನೈದು": 15,
            "ಹದಿನಾರು": 16, "ಹದಿನೇಳು": 17, "ಹದಿನೆಂಟು": 18, "ಹತ್ತೊಂಬತ್ತು": 19}
KN_TENS = {"ಇಪ್ಪತ್ತು": 20, "ಮೂವತ್ತು": 30, "ನಲವತ್ತು": 40, "ಐವತ್ತು": 50, "ಅರವತ್ತು": 60, "ಎಪ್ಪತ್ತು": 70,
           "ಎಂಬತ್ತು": 80, "ತೊಂಬತ್ತು": 90}
KN_MULT = {"ನೂರು": 100, "ಸಾವಿರ": 1000, "ಲಕ್ಷ": 100000, "ಕೋಟಿ": 10 ** 7}
# regular compounds: tens word minus the final vowel sign "ು" + a sandhi form of the unit (ಇಪ್ಪತ್ತೈದು = 25)
KN_COMPOUND_UNITS = {1: "ೊಂದು", 2: "ೆರಡು", 3: "ಮೂರು", 4: "ನಾಲ್ಕು", 5: "ೈದು", 6: "ಾರು", 7: "ೇಳು", 8: "ೆಂಟು",
                     9: "ೊಂಬತ್ತು"}

_add(EN_NUM, _en_kind)
_add(EN_ORD, _en_kind, ordinal=True)
_add(HI_NUM, _hi_kind)
_add(ML_UNITS, _ind_kind); _add(ML_TEENS, _ind_kind); _add(ML_TENS, _ind_kind)
for _stem, _t in ML_COMPOUND_STEMS.items():
    for _u, _suffix in ML_COMPOUND_UNITS.items():
        NUMBER_WORDS[fold(_stem + _suffix)] = (_t + _u, "whole", False)
_add(KN_UNITS, _ind_kind); _add(KN_TEENS, _ind_kind); _add(KN_TENS, _ind_kind)
for _w, _t in KN_TENS.items():
    for _u, _suffix in KN_COMPOUND_UNITS.items():
        NUMBER_WORDS[fold(_w[:-1] + _suffix)] = (_t + _u, "whole", False)

MULTIPLIERS: Dict[str, int] = {}
for _tbl in ({"hundred": 100, "thousand": 1000, "lakh": 10 ** 5, "lakhs": 10 ** 5, "lac": 10 ** 5, "lacs": 10 ** 5,
              "crore": 10 ** 7, "crores": 10 ** 7, "million": 10 ** 6},
             {"सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 10 ** 5, "करोड़": 10 ** 7, "करोड": 10 ** 7},
             ML_MULT, KN_MULT):
    for _w, _v in _tbl.items():
        MULTIPLIERS[fold(_w)] = _v
        NUMBER_WORDS[fold(_w)] = (_v, "mult", False)

# digit-by-digit speech ("nine eight seven ...", "double five")
DIGIT_EXTRA = {"oh": 0, "o": 0}                      # English "oh" for zero: only inside a run of digits
DOUBLE_WORDS = {fold(w): n for w, n in {"double": 2, "triple": 3, "डबल": 2, "ट्रिपल": 3, "ഡബിൾ": 2,
                                        "ട്രിപ്പിൾ": 3, "ಡಬಲ್": 2, "ಟ್ರಿಪಲ್": 3}.items()}

# --------------------------------------------------------------------------- months
_MONTHS_SRC = {
    1: ["january", "jan", "जनवरी", "ജനുവരി", "ಜನವರಿ"],
    2: ["february", "feb", "फरवरी", "फ़रवरी", "ഫെബ്രുവരി", "ಫೆಬ್ರವರಿ"],
    3: ["march", "mar", "मार्च", "മാർച്ച്", "ಮಾರ್ಚ್"],
    4: ["april", "apr", "अप्रैल", "अप्रेल", "ഏപ്രിൽ", "ಏಪ್ರಿಲ್"],
    5: ["may", "मई", "മേയ്", "മെയ്", "ಮೇ"],
    6: ["june", "jun", "जून", "ജൂൺ", "ಜೂನ್"],
    7: ["july", "jul", "जुलाई", "ജൂലൈ", "ಜುಲೈ"],
    8: ["august", "aug", "अगस्त", "ഓഗസ്റ്റ്", "ആഗസ്റ്റ്", "ಆಗಸ್ಟ್"],
    9: ["september", "sept", "sep", "सितंबर", "सितम्बर", "സെപ്റ്റംബർ", "ಸೆಪ್ಟೆಂಬರ್"],
    10: ["october", "oct", "अक्टूबर", "अक्तूबर", "ഒക്ടോബർ", "ಅಕ್ಟೋಬರ್"],
    11: ["november", "nov", "नवंबर", "नवम्बर", "നവംബർ", "ನವೆಂಬರ್"],
    12: ["december", "dec", "दिसंबर", "दिसम्बर", "ഡിസംബർ", "ಡಿಸೆಂಬರ್"],
}
MONTH_WORDS: Dict[str, int] = {fold(w): m for m, ws in _MONTHS_SRC.items() for w in ws}
# Indic month names take case endings ("ജനുവരിയിൽ", "मार्च को"): allow a prefix match for long, non-Latin stems
MONTH_PREFIXES: List[Tuple[str, int]] = [(w, m) for w, m in MONTH_WORDS.items() if len(w) >= 5 and not w.isascii()]
MONTH_NAMES_EN = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September",
                  "October", "November", "December"]
MONTH_NAMES = {   # for reading a date back in the user's language
    "en": MONTH_NAMES_EN,
    "hi": ["", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "ml": ["", "ജനുവരി", "ഫെബ്രുവരി", "മാർച്ച്", "ഏപ്രിൽ", "മേയ്", "ജൂൺ", "ജൂലൈ", "ഓഗസ്റ്റ്", "സെപ്റ്റംബർ", "ഒക്ടോബർ",
           "നവംബർ", "ഡിസംബർ"],
    "kn": ["", "ಜನವರಿ", "ಫೆಬ್ರವರಿ", "ಮಾರ್ಚ್", "ಏಪ್ರಿಲ್", "ಮೇ", "ಜೂನ್", "ಜುಲೈ", "ಆಗಸ್ಟ್", "ಸೆಪ್ಟೆಂಬರ್", "ಅಕ್ಟೋಬರ್",
           "ನವೆಂಬರ್", "ಡಿಸೆಂಬರ್"],
}

RELATIVE_DAYS = {"today": 0, "tomorrow": 1, "yesterday": -1}          # English only (Hindi "कल" is ambiguous)

# --------------------------------------------------------------------------- gender
GENDER_WORDS = {
    "Male": _f(["male", "man", "boy", "gentleman", "m", "पुरुष", "मर्द", "आदमी", "लड़का", "പുരുഷൻ", "ആൺ",
                "ആൺകുട്ടി", "ಪುರುಷ", "ಗಂಡು", "ಹುಡುಗ"]),
    "Female": _f(["female", "woman", "girl", "lady", "f", "महिला", "औरत", "स्त्री", "लड़की", "സ്ത്രീ", "പെൺ",
                  "പെൺകുട്ടി", "ಮಹಿಳೆ", "ಹೆಣ್ಣು", "ಸ್ತ್ರೀ", "ಹುಡುಗಿ"]),
    "Other": _f(["other", "transgender", "trans", "nonbinary", "non-binary", "अन्य", "ट्रांसजेंडर", "ട്രാൻസ്ജെൻഡർ",
                 "മറ്റുള്ളവ", "ಇತರೆ", "ಟ್ರಾನ್ಸ್ಜೆಂಡರ್"]),
}

# --------------------------------------------------------------------------- blood groups
BLOOD_LETTERS = {"a": "A", "b": "B", "o": "O", "ab": "AB", "ए": "A", "बी": "B", "ओ": "O", "एबी": "AB",
                 "എ": "A", "ബി": "B", "ഒ": "O", "ഓ": "O", "എബി": "AB", "ಎ": "A", "ಬಿ": "B", "ಓ": "O", "ಎಬಿ": "AB"}
BLOOD_POS = _f(["positive", "plus", "pos", "+ve", "पॉजिटिव", "पॉज़िटिव", "पाजिटिव", "पॉजीटिव", "प्लस", "പോസിറ്റീവ്",
                "പ്ലസ്", "ಪಾಸಿಟಿವ್", "ಪ್ಲಸ್"])
BLOOD_NEG = _f(["negative", "minus", "neg", "-ve", "नेगेटिव", "नेगटिव", "माइनस", "നെഗറ്റീവ്", "മൈനസ്", "ನೆಗೆಟಿವ್",
                "ಮೈನಸ್"])
BLOOD_SPOKEN = {"en": {"+": "positive", "-": "negative"}, "hi": {"+": "पॉजिटिव", "-": "नेगेटिव"},
                "ml": {"+": "പോസിറ്റീവ്", "-": "നെഗറ്റീവ്"}, "kn": {"+": "ಪಾಸಿಟಿವ್", "-": "ನೆಗೆಟಿವ್"}}

# --------------------------------------------------------------------------- confirmations
YES_WORDS = _f([
    "yes", "yeah", "yep", "yup", "ya", "yea", "correct", "right", "exactly", "ok", "okay", "alright", "sure",
    "absolutely", "definitely", "affirmative", "confirmed", "confirm", "true", "fine", "proceed", "continue",
    "हाँ", "हां", "हा", "जी", "सही", "ठीक", "बिलकुल", "बिल्कुल", "सही है", "ठीक है", "जी हाँ", "जी हां", "हाँ जी", "हां जी",
    "അതെ", "ശരി", "ശരിയാണ്", "ഉവ്വ്", "ഓകെ", "ശരിയാണു", "ഷുവർ",
    "ಹೌದು", "ಸರಿ", "ಸರಿಯಾಗಿದೆ", "ಓಕೆ", "ಹೌದು ಸರಿ",
])
NO_WORDS = _f([
    "no", "nope", "nah", "nay", "incorrect", "wrong", "not correct", "not right", "false", "negative",
    "नहीं", "नही", "नहिं", "गलत", "ग़लत", "सही नहीं", "ठीक नहीं", "सही नही",
    "അല്ല", "ഇല്ല", "തെറ്റ്", "തെറ്റാണ്", "ശരിയല്ല", "ശരിയല്ലാ", "അല്ലാ",
    "ಇಲ್ಲ", "ಅಲ್ಲ", "ತಪ್ಪು", "ತಪ್ಪಾಗಿದೆ", "ಸರಿಯಲ್ಲ", "ಸರಿ ಅಲ್ಲ",
])
RETRY_WORDS = _f([
    "say again", "say it again", "say that again", "again", "repeat", "repeat that", "repeat it", "once more", "one more time",
    "pardon", "come again", "try again", "i did not hear", "i didn't hear", "did not catch", "didn't catch", "what",
    "can you repeat", "please repeat", "read again", "tell again", "start over", "re record", "record again",
    "फिर से", "दोबारा", "दुबारा", "फिर बोलो", "फिर बोलिए", "एक बार और", "दोहराओ", "दोहराइए", "फिर से बोलिए", "फिर से बोलो",
    "ഒന്നുകൂടി", "ഒന്ന് കൂടി", "വീണ്ടും", "ആവർത്തിക്കൂ", "ഒന്നുകൂടി പറയൂ",
    "ಮತ್ತೆ", "ಇನ್ನೊಮ್ಮೆ", "ಮತ್ತೊಮ್ಮೆ", "ಮತ್ತೆ ಹೇಳಿ", "ಇನ್ನೊಮ್ಮೆ ಹೇಳಿ",
])
# "not correct", "isn't right", "was not ok" -> no  (English negation comes BEFORE the positive word)
ENGLISH_POSITIVE_FOR_NEGATION = _f(["correct", "right", "ok", "okay", "fine", "true", "sure", "good", "accurate"])
NEGATORS_EN = _f(["not", "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent", "never", "hardly"])
# words that carry no new information when judging whether a "no, ..." reply contains a correction
CORRECTION_STOPWORDS = _f([
    "it", "its", "it's", "is", "was", "that", "that's", "thats", "this", "the", "a", "an", "please", "sorry", "wait",
    "actually", "well", "so", "um", "uh", "er", "hmm", "oh", "i", "said", "mean", "meant", "no", "not", "wrong",
    "incorrect", "right", "correct", "again", "say", "it's", "there", "mistake", "है", "वो", "यह", "ये", "तो", "अरे",
    "ആണ്", "ഇത്", "അത്", "ആ", "ಅದು", "ಇದು", "ಆಗಿದೆ",
])

# --------------------------------------------------------------------------- fillers and lead-in phrases
FILLERS = _f(["um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "hm", "ah", "ahh", "aah", "eh", "अं", "अम्म", "ഉം", "ಉಂ"])
TITLES = _f(["mr", "mrs", "ms", "miss", "shri", "shree", "sri", "smt", "श्री", "श्रीमती", "श्रीमान", "ശ്രീ", "ശ്രീമതി",
             "ಶ್ರೀ", "ಶ್ರೀಮತಿ"])
GREETINGS = _f(["hello", "hi", "hey", "namaste", "namaskar", "good morning", "good afternoon", "good evening",
                "नमस्ते", "नमस्कार", "നമസ്കാരം", "ನಮಸ್ಕಾರ", "sir", "madam", "please", "ok", "okay"])
NAME_LEAD = _f([
    "my full name is", "my name is", "my name's", "the name is", "name is", "this is", "i am", "i'm", "it is", "it's",
    "call me", "they call me", "you can call me", "my name", "name",
    "मेरा पूरा नाम", "मेरा नाम", "मेरा नाम है", "पूरा नाम", "नाम", "मैं", "मेरे को",
    "എന്റെ പേര്", "എന്റെ പേരു", "എൻ്റെ പേര്", "പേര്", "ഞാൻ",
    "ನನ್ನ ಹೆಸರು", "ಹೆಸರು", "ನಾನು",
])
NAME_TRAIL = _f(["is my name", "please", "thank you", "thanks", "है", "हूँ", "हूं", "हैं", "ആണ്", "എന്നാണ്", "ആകുന്നു",
                 "ആണു", "ಆಗಿದೆ", "ಎಂದು", "ಎಂಬುದು", "ಇದೆ"])
ADDRESS_LEAD = _f([
    "my address is", "the address is", "address is", "my full address is", "i live at", "i live in", "i stay at",
    "i stay in", "i am staying at", "i'm staying at", "i reside at", "i reside in", "residing at", "my house is at",
    "my house is", "i am from", "i'm from", "i belong to", "it is", "it's", "my address",
    "मेरा पता", "मेरा पता है", "पता", "मैं रहता हूँ", "मैं रहता हूं", "मैं रहती हूँ", "मैं रहती हूं", "मेरा घर",
    "എന്റെ വിലാസം", "വിലാസം", "ഞാൻ താമസിക്കുന്നത്", "ഞാൻ താമസിക്കുന്നു",
    "ನನ್ನ ವಿಳಾಸ", "ವಿಳಾಸ", "ನಾನು ವಾಸಿಸುತ್ತಿದ್ದೇನೆ",
])
ADDRESS_TRAIL = _f(["is my address", "please", "thank you", "है", "हूँ", "हूं", "ആണ്", "ಆಗಿದೆ"])
TEXT_LEAD = _f(["the problem is", "my problem is", "the issue is", "my issue is", "the reason is", "the reason for visit is",
                "i have come for", "i came for", "i work as a", "i work as an", "i work as", "i am working as a",
                "i am working as an", "my occupation is", "my job is", "i am a", "i am an", "it is", "it's"])
TEXT_TRAIL = _f(["thank you", "thanks", "please", "that is all", "that's all", "that is it", "that's it", "धन्यवाद", "शुक्रिया",
                 "നന്ദി", "ಧನ್ಯವಾದಗಳು"])
EMAIL_STOPWORDS = _f(["um", "umm", "uh", "uhh", "er", "erm", "hmm", "my", "email", "e-mail", "mail", "id", "address", "is", "it",
                      "its", "it's", "the", "please", "thanks", "thank", "you", "hello", "hi", "okay", "ok", "so", "and", "yes",
                      "that", "this", "ईमेल", "मेरा", "है", "ഇമെയിൽ", "ಇಮೇಲ್"])
ADDRESS_NUMBER_MARKERS = _f(["no", "number", "door", "house", "flat", "plot", "h", "नंबर", "नं", "मकान", "ഡോർ", "നമ്പർ",
                             "ಸಂಖ್ಯೆ", "ನಂಬರ್"])
AGE_UNITS = {"years": _f(["year", "years", "yr", "yrs", "yo", "साल", "वर्ष", "बरस", "വയസ്", "വയസ്സ്", "വയസ്സാണ്",
                          "വയസ്സായി", "ವರ್ಷ", "ವರ್ಷಗಳು", "ವಯಸ್ಸು"]),
             "months": _f(["month", "months", "महीने", "महीना", "മാസം", "മാസങ്ങൾ", "ತಿಂಗಳು", "ತಿಂಗಳ"])}

# --------------------------------------------------------------------------- spoken e-mail
EMAIL_AT = _f(["at", "at the rate", "at the rate of", "एट द रेट", "एट", "അറ്റ്", "ಅಟ್"])
EMAIL_DOT = _f(["dot", "डॉट", "ഡോട്ട്", "ಡಾಟ್", "point"])
EMAIL_SYMBOLS = {fold(k): v for k, v in {"underscore": "_", "dash": "-", "hyphen": "-", "अंडरस्कोर": "_", "डैश": "-"}.items()}

# --------------------------------------------------------------------------- field labels (for messages / read-back)
FIELD_LABELS = {  # english label per canonical field; other languages are produced by the dialogue (3A)
    "name": "name", "age": "age", "gender": "gender", "phone": "phone number", "date_of_birth": "date of birth",
    "date": "date", "address": "address", "pincode": "PIN code", "aadhaar": "Aadhaar number",
    "blood_group": "blood group", "email": "email address", "complaint": "problem", "occupation": "occupation",
    "yes_no": "answer", "amount": "amount",
}
