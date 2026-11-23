"""
publish.py -- template -> voice-call form, and call answers -> values for filling
================================================================================
    form_spec = publish_template(template, FormStore("data/forms"))        # the call now asks this form's questions
    values    = answers_from_record(call_record)                          # {call_name: final value}
    pdf, rep  = fill_from_record(template, pdf_bytes, call_record)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import extractor
from callpage.forms import FormError, FormSpec, FormStore, build_form_dict
from formreader import semantics
from formreader.fill import fill_pdf
from formreader.model import FormTemplate


def form_dict(template: FormTemplate, languages=("en",)) -> Dict[str, Any]:
    fields = []
    for f in template.active_fields():
        fields.append({"name": f.call_name, "kind": f.call_kind if f.call_kind in extractor.RULES else "text",
                       "required": f.required, "label": {"en": f.label}, "question": {"en": semantics.make_question(f)}})
    return build_form_dict(template.title, fields, "en", list(languages), form_id=template.id)


def publish_template(template: FormTemplate, store: FormStore) -> FormSpec:
    """Save the call form (form id = template id).  Raises ``FormError`` with a readable message."""
    template.validate()
    if not template.active_fields():
        raise FormError("every field is ignored: nothing to ask")
    spec = store.save(form_dict(template))
    # the extractor may map a name to its canonical one (alias): keep the template in step so filling finds the answers
    canon = {f.name for f in spec.fields}
    for f in template.active_fields():
        try:
            c = extractor.resolve_field(f.call_name).name
        except extractor.UnknownFieldError:
            c = f.call_name
        if c != f.call_name and c in canon:
            f.call_name = c
    return spec


def answers_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Final answer per field of a stored call: confirmed values; a later correction replaces the earlier value.
    ``record["corrections"]`` may be a dict {name: value} or a list of {"field"/"name": ..., "value": ...} (last wins)."""
    out: Dict[str, Any] = {}
    for f in record.get("fields", []):
        if f.get("status") == "confirmed" and f.get("value") not in (None, ""):
            out[f["name"]] = f["value"]
    corr = record.get("corrections") or {}
    if isinstance(corr, dict):
        items = corr.items()
    else:
        items = []
        for c in (corr if isinstance(corr, list) else []):
            if not isinstance(c, dict):
                continue
            name = c.get("field") or c.get("name")
            value = c.get("new")
            if value is None:
                value = c.get("value")
            items.append((name, value))
    for name, value in items:
        if name and value not in (None, ""):
            out[name] = value
    return out


def fill_from_record(template: FormTemplate, pdf_bytes: bytes, record: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    return fill_pdf(template, pdf_bytes, answers_from_record(record))
