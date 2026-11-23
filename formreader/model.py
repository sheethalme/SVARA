"""
model.py -- the form TEMPLATE: what the reader found on the uploaded PDF
=========================================================================

Coordinates are in PDF points as reported by pdfplumber: ``x`` is the PDF x, ``top`` is ``page_height - y`` (so y grows
DOWN the page).  A box is ``(x0, top, x1, bottom)``.  ``page_height`` is stored per page so filling can convert back.

A field has an input SHAPE (``shape``: text / multiline / comb / checkbox / choice / signature / photo) and a MEANING
(``call_kind`` / ``call_name``: which question the voice call asks and how the answer is understood, using the same
kinds as the extractor: name, age, gender, phone, date_of_birth, address, pincode, aadhaar, blood_group, email, text,
number, yes_no ...).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]
SHAPES = ("text", "multiline", "comb", "checkbox", "choice", "signature", "photo")
_NAME = re.compile(r"^[a-z0-9_]{1,48}$")


class TemplateError(ValueError):
    """The template is invalid (the message says what to fix)."""


@dataclass
class Option:
    label: str
    box: Box                                 # the tick box of this option


@dataclass
class TField:
    id: str
    label: str
    page: int                                # 0-based
    bbox: Box
    shape: str = "text"
    call_kind: str = "text"                  # extractor kind used to understand the spoken answer
    call_name: str = ""                      # field name in the voice-call form
    required: bool = True
    ignored: bool = False                    # not asked in the call (signature, office-use area ...)
    options: List[Option] = field(default_factory=list)
    cells: List[Box] = field(default_factory=list)          # comb: one box per character
    line_boxes: List[Box] = field(default_factory=list)     # multiline: one box per written line
    date_format: Optional[str] = None        # "DD/MM/YYYY" ...
    question: Optional[str] = None           # custom question (else generated)
    source: str = ""                         # acroform | underline | box | dots | table | comb | tick
    confidence: float = 0.5
    notes: List[str] = field(default_factory=list)
    acro_name: Optional[str] = None          # name of the fillable field (AcroForm PDFs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TField":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        d["options"] = [Option(o["label"], tuple(o["box"])) for o in d.get("options", [])]
        d["cells"] = [tuple(c) for c in d.get("cells", [])]
        d["line_boxes"] = [tuple(c) for c in d.get("line_boxes", [])]
        return cls(**d)


@dataclass
class PageInfo:
    width: float
    height: float
    rotation: int = 0


@dataclass
class FormTemplate:
    id: str
    title: str
    source_name: str
    pages: List[PageInfo]
    fields: List[TField] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    acroform: bool = False
    created_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "source_name": self.source_name, "acroform": self.acroform,
                "created_at": self.created_at, "warnings": self.warnings,
                "pages": [asdict(p) for p in self.pages], "fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormTemplate":
        try:
            t = cls(d["id"], d["title"], d.get("source_name", ""), [PageInfo(**p) for p in d["pages"]],
                    [TField.from_dict(f) for f in d["fields"]], list(d.get("warnings", [])), bool(d.get("acroform")),
                    d.get("created_at", ""))
        except (KeyError, TypeError) as exc:
            raise TemplateError(f"the template is damaged ({type(exc).__name__}: {exc})") from None
        t.validate()
        return t

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def active_fields(self) -> List[TField]:
        return [f for f in self.fields if not f.ignored]

    def validate(self) -> None:
        ids = set()
        for f in self.fields:
            if f.id in ids:
                raise TemplateError(f"field id '{f.id}' appears twice")
            ids.add(f.id)
            if f.shape not in SHAPES:
                raise TemplateError(f"field '{f.label}': unknown shape '{f.shape}'")
            if not 0 <= f.page < len(self.pages):
                raise TemplateError(f"field '{f.label}': page {f.page + 1} does not exist")
            x0, t, x1, b = f.bbox
            if not (x1 > x0 and b > t):
                raise TemplateError(f"field '{f.label}': the box is empty")
            if not f.ignored and not _NAME.match(f.call_name or ""):
                raise TemplateError(f"field '{f.label}': call name '{f.call_name}' must be lower-case letters, digits and _")
