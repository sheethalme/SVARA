"""
semantics.py -- understanding the LABELS of a form with plain rules (no AI, nothing leaves the computer)
=========================================================================================================

* ``clean_label``     "1. Date of Birth (dd/mm/yyyy) *:"  ->  label "Date of Birth", required, date format
* ``infer_kind``      which kind of answer the label asks for ("Mobile No." -> phone, "Name of the Applicant" -> name)
* ``assign_names``    unique call-field names ("name", "fathers_name", "permanent_address" ...)
* ``make_question``   the English question the voice call will ask

English only (phase 1).  Every rule is a visible regular expression, so a wrong guess can be corrected in the review screen
and the rules can be improved from real forms.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from formreader.model import TField

# ------------------------------------------------------------------------------------------------ cleaning
_NUMBERING = re.compile(r"^\s*(?:\(?\d{1,2}[\).:\-]|\(?[a-zA-Z][\).]\s|\(?[ivxIVX]{1,4}[\).])\s*")
_DATE_FORMATS = [(re.compile(r"d{1,2}\s*[/\-.]\s*m{1,2}\s*[/\-.]\s*y{4}", re.I), None),
                 (re.compile(r"y{4}\s*[/\-.]\s*m{1,2}\s*[/\-.]\s*d{1,2}", re.I), None),
                 (re.compile(r"m{1,2}\s*[/\-.]\s*d{1,2}\s*[/\-.]\s*y{2,4}", re.I), None),
                 (re.compile(r"d{1,2}\s*[/\-.]\s*m{1,2}\s*[/\-.]\s*y{2}", re.I), None)]
_OPTIONAL = re.compile(r"\b(optional|if any|if applicable|n/?a)\b", re.I)


def detect_date_format(text: str) -> Optional[str]:
    """"dd/mm/yyyy" -> "DD/MM/YYYY" (separator kept).  None when the text does not show a format."""
    for rx, _ in _DATE_FORMATS:
        m = rx.search(text or "")
        if m:
            return re.sub(r"\s+", "", m.group(0)).upper()
    return None


def _smart_case(s: str) -> str:
    if len(s) > 3 and s.isupper():
        return re.sub(r"(^|[\s/(-])([a-z])", lambda m: m.group(1) + m.group(2).upper(), s.lower())
    return s


def clean_label(raw: str) -> Tuple[str, bool, Dict[str, object]]:
    """Return ``(label, required, hints)``.  ``required`` is True unless the label says optional;
    a trailing ``*`` is the usual "required" mark (it also keeps required = True)."""
    s = re.sub(r"\s+", " ", raw or "").strip()
    hints: Dict[str, object] = {}
    fmt = detect_date_format(s)
    if fmt:
        hints["date_format"] = fmt
    required = True
    if _OPTIONAL.search(s):
        required = False
    s = re.sub(r"\([^)]*\)", " ", s)                             # (dd/mm/yyyy), (in block letters), (Male/Female)
    s = s.replace("*", " ")
    s = _NUMBERING.sub("", s)
    s = re.sub(r"[\s:;.\-–—_…]+$", "", s)
    s = re.sub(r"^[\s:;.\-–—_…]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _smart_case(s), required, hints


# ------------------------------------------------------------------------------------------------ meaning
_KIND_RULES: List[Tuple[re.Pattern, str]] = [(re.compile(p, re.I), k) for p, k in [
    (r"\bsignature\b|\bsign\b|thumb|\bseal\b|\bstamp\b", "signature"),
    (r"\bphoto(graph)?\b|passport[- ]size", "photo"),
    (r"e-?\s?mail", "email"),
    (r"aadhaa?r|\buid\b", "aadhaar"),
    (r"\bpin\b|pin\s?code|postal\s+code|\bzip\b", "pincode"),
    (r"mobile|phone|\bcontact\b|\btel\b|whats\s?app|\bcell\b", "phone"),
    (r"date\s+of\s+birth|\bd\.?\s?o\.?\s?b\b|birth\s?date|born\s+on", "date_of_birth"),
    (r"blood\s+(group|type)", "blood_group"),
    (r"\bgender\b|\bsex\b", "gender"),
    (r"\bage\b", "age"),
    (r"\baddress\b", "address"),
    (r"\bname\b", "name"),
    (r"annual|monthly|income|salary|amount|\bfees?\b", "number"),
    (r"\bdate\b|\bdated\b", "date"),
    (r"\b(do|are|have|whether|is)\s+(you|your|the)\b|\byes\b.*\bno\b|\by\s*/\s*n\b", "yes_no"),
]]
_NOT_A_PERSON_NAME = re.compile(r"name\s+of\s+(the\s+)?(school|college|institution|institute|university|board|company|employer|"
                                r"organi[sz]ation|hospital|bank|branch|course|programme|program|scheme|department|"
                                r"examination|exam|doctor|insurer|policy|village|town|city|state|district|country)", re.I)
_QUALIFIER = re.compile(r"\b(father|mother|guardian|spouse|husband|wife|parent|parents|emergency|permanent|present|current|"
                        r"correspondence|local|previous|office|nominee|referee|relative|sibling|brother|sister)\b", re.I)
CANONICAL = ("name", "age", "gender", "phone", "date_of_birth", "address", "pincode", "aadhaar", "blood_group", "email")


def infer_kind(label: str, shape: str = "text", comb_len: Optional[int] = None) -> str:
    """The extractor kind a label asks for.  Unknown labels are plain ``text``."""
    low = (label or "").lower()
    kind = "text"
    for rx, k in _KIND_RULES:
        if rx.search(low):
            kind = k
            break
    if kind == "name" and _NOT_A_PERSON_NAME.search(low):
        kind = "text"
    if kind in ("number",) and re.search(r"percent|marks|cgpa|grade|score|%", low):
        kind = "text"
    if shape == "comb" and kind == "text" and comb_len:
        kind = {12: "aadhaar", 10: "phone", 6: "pincode"}.get(comb_len, "number")
    if shape == "checkbox" and kind == "text":
        kind = "yes_no"
    if kind in ("signature", "photo"):
        return kind
    return kind


def has_qualifier(label: str) -> bool:
    return bool(_QUALIFIER.search(label or ""))


def slugify_name(label: str) -> str:
    s = re.sub(r"['’`]", "", (label or "").lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:40].strip("_")
    return ("f_" + s) if s and s[0].isdigit() else (s or "field")


def assign_names(fields: List[TField]) -> None:
    """Set ``call_kind`` and a unique ``call_name`` for every non-ignored field (reading order matters:
    the first plain "Name" becomes ``name``, "Father's Name" becomes ``fathers_name`` ...)."""
    used = set()
    for f in fields:
        if f.ignored:
            continue
        kind = infer_kind(f.label, f.shape, len(f.cells) or None)
        f.call_kind = kind
        if kind in CANONICAL and not has_qualifier(f.label) and kind not in used:
            name = kind
        else:
            name = slugify_name(f.label)
            if kind in CANONICAL and name in CANONICAL and name != kind:
                name = f"{name}_field"
        base, i = name, 2
        while name in used:
            name = f"{base}_{i}"; i += 1
        used.add(name)
        f.call_name = name


# ------------------------------------------------------------------------------------------------ questions
def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


def make_question(f: TField) -> str:
    """The English question the call asks for this field (``f.question`` wins when the admin wrote one)."""
    if f.question:
        return f.question
    from callpage import prompts
    label = (f.label or "this").strip()
    low = label.lower()
    opts = [o.label for o in f.options]
    if f.shape == "choice" and opts:
        joined = ", ".join(opts[:-1]) + (" or " if len(opts) > 1 else "") + opts[-1]
        if f.call_kind == "gender":
            return f"What is your gender? {joined}?"
        return f"Which {low}? {joined}?"
    if f.shape == "checkbox" or f.call_kind == "yes_no":
        if re.match(r"^i\s", low):
            return f'The form says: "{label}". Do you agree? Please say yes or no.'
        return f"{label}? Please say yes or no."
    canon = prompts.QUESTIONS.get(f.call_name)
    if canon and f.call_name in CANONICAL and f.call_kind == f.call_name:
        return canon["en"]
    if f.call_kind in ("date", "date_of_birth"):
        return f"What is the {low}? Please say the day, month and year."
    if low.startswith("name of"):
        return f"What is the {low}?"
    if f.shape == "comb" or f.call_kind in ("number",):
        extra = " Please say the digits one by one." if f.shape == "comb" else ""
        return f"What is your {low}?{extra}"
    if re.match(r"^(your|the)\s", low):
        return f"What is {low}?"
    return f"What is your {low}?"
