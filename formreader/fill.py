"""
fill.py -- writes the confirmed answers onto the ORIGINAL PDF (locally)
======================================================================
    pdf_bytes, report = fill_pdf(template, source_pdf, {"name": "Anita Rao", "gender": "Female", ...})

``values`` is keyed by the call field name (``TField.call_name``).  Fillable (AcroForm) PDFs get their real fields set;
flat PDFs get text drawn on a transparent layer that is merged over each page, at the box found by the analysis.
The report lists what was written, skipped and why -- nothing is dropped silently.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import io
import re
from typing import Any, Dict, List, Optional, Tuple

from formreader.model import Box, FormTemplate, TField

FONT = "Helvetica"
_YES = {"yes", "y", "true", "1", "agree", "agreed", "checked", "ok", "yeah", "yep"}


def format_date(value: str, fmt: Optional[str]) -> str:
    """ISO ``2001-03-05`` -> the form's format (default DD/MM/YYYY).  Other text is returned unchanged."""
    m = re.fullmatch(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*", value or "")
    if not m:
        return value
    y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
    fmt = (fmt or "DD/MM/YYYY").upper()
    sep = next((c for c in fmt if c in "/-."), "")
    if fmt in ("DDMMYYYY",):
        return d + mo + y
    if fmt in ("DDMMYY",):
        return d + mo + y[2:]
    parts = re.findall(r"D+|M+|Y+", fmt)
    pieces = {"D": d, "M": mo, "Y": y if len(max(parts, key=lambda p: len(p) if p[0] == "Y" else 0)) == 4 else y[2:]}
    out = []
    for p in parts:
        out.append(y[2:] if p[0] == "Y" and len(p) == 2 else pieces[p[0]])
    return sep.join(out)


def _latin(text: str, warn: List[str], label: str) -> str:
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        kept = text.encode("latin-1", "ignore").decode("latin-1")
        warn.append(f"'{label}': characters outside the PDF's standard font were left out ({text!r} -> {kept!r})")
        return kept


def _match_option(value: str, field: TField) -> Optional[int]:
    v = (value or "").strip().lower()
    labels = [o.label.strip().lower() for o in field.options]
    if v in labels:
        return labels.index(v)
    for i, l in enumerate(labels):
        if v and (v in l or l in v):
            return i
    near = difflib.get_close_matches(v, labels, n=1, cutoff=0.7)
    if near:
        return labels.index(near[0])
    if field.call_kind == "gender":                               # "M"/"F"
        for i, l in enumerate(labels):
            if v[:1] and l.startswith(v[:1]):
                return i
    return None


def _wrap(c, text: str, font: str, size: float, width: float) -> List[str]:
    lines: List[str] = []
    cur = ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if c.stringWidth(trial, font, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_text(c, text: str, box: Box, ph: float, ox: float, oy: float, underline: bool) -> bool:
    """Draw ``text`` in ``box`` (top-based).  Returns False if it had to be cut."""
    x0, top, x1, bot = box
    width = x1 - x0 - 3
    size = 11.0 if (bot - top) >= 13 else max(7.0, (bot - top) - 2)
    while size > 7 and c.stringWidth(text, FONT, size) > width:
        size -= 0.5
    cut = False
    while text and c.stringWidth(text, FONT, size) > width:
        text = text[:-1]; cut = True
    base = bot - 2.5 if underline else (top + bot) / 2 - size * 0.32
    c.setFont(FONT, size)
    c.drawString(ox + x0 + 1.5, oy + ph - base, text)
    return not cut


def _draw_multiline(c, text: str, f: TField, ph: float, ox: float, oy: float) -> bool:
    boxes = f.line_boxes or [f.bbox]
    if len(boxes) > 1:                                                 # written lines: fill line by line
        size = 10.0
        width = min(b[2] - b[0] for b in boxes) - 3
        lines = _wrap(c, text, FONT, size, width)
        while len(lines) > len(boxes) and size > 7:
            size -= 0.5; lines = _wrap(c, text, FONT, size, width)
        c.setFont(FONT, size)
        for ln, b in zip(lines, boxes):
            c.drawString(ox + b[0] + 1.5, oy + ph - (b[3] - 2.5), ln)
        return len(lines) <= len(boxes)
    x0, top, x1, bot = f.bbox
    size, ok = 10.0, True
    width = x1 - x0 - 6
    lines = _wrap(c, text, FONT, size, width)
    while size > 7 and (len(lines) * (size + 2) > (bot - top) - 4):
        size -= 0.5; lines = _wrap(c, text, FONT, size, width)
    fit = max(1, int(((bot - top) - 4) // (size + 2)))
    if len(lines) > fit:
        lines, ok = lines[:fit], False
    c.setFont(FONT, size)
    for i, ln in enumerate(lines):
        c.drawString(ox + x0 + 3, oy + ph - (top + 3 + size + i * (size + 2) - 1), ln)
    return ok


def _draw_comb(c, text: str, f: TField, ph: float, ox: float, oy: float) -> bool:
    chars = re.sub(r"[\s/-]", "", text) if f.call_kind in ("aadhaar", "phone", "pincode", "number") or f.date_format else text
    cells = f.cells
    c.setFont(FONT, 11)
    for ch, (x0, top, x1, bot) in zip(chars, cells):
        c.drawCentredString(ox + (x0 + x1) / 2, oy + ph - ((top + bot) / 2 + 3.6), ch)
    return len(chars) <= len(cells)


def _tick(c, box: Box, ph: float, ox: float, oy: float) -> None:
    x0, top, x1, bot = box
    size = max(6.0, min(x1 - x0, bot - top) + 1)
    c.setFont("ZapfDingbats", size)
    c.drawCentredString(ox + (x0 + x1) / 2, oy + ph - (bot - (bot - top - size * 0.7) / 2 - 0.4) + 0.0, "4")


def _text_for(f: TField, value: Any, warn: List[str]) -> str:
    text = str(value).strip()
    if f.call_kind in ("date", "date_of_birth"):
        text = format_date(text, f.date_format)
    return _latin(text, warn, f.label)


def fill_pdf(template: FormTemplate, source, values: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Return ``(filled_pdf_bytes, report)``.  ``source`` is the blank PDF (path or bytes)."""
    from pypdf import PdfReader, PdfWriter

    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
    report: Dict[str, Any] = {"filled": [], "skipped": [], "warnings": []}
    warn: List[str] = report["warnings"]
    reader = PdfReader(io.BytesIO(raw))
    writer = PdfWriter(clone_from=reader)
    todo: List[Tuple[TField, Any]] = []
    for f in template.fields:
        if f.ignored:
            continue
        v = values.get(f.call_name)
        if v is None or str(v).strip() == "":
            report["skipped"].append({"field": f.call_name, "label": f.label, "why": "no answer"})
            continue
        todo.append((f, v))
    if any(int(p.get("/Rotate", 0) or 0) for p in reader.pages):
        warn.append("The PDF has rotated pages: written text may be misplaced. Check the result.")

    from reportlab.pdfgen import canvas

    overlays: Dict[int, Any] = {}
    bufs: Dict[int, io.BytesIO] = {}

    def canv(pno: int):
        if pno not in overlays:
            mb = reader.pages[pno].mediabox
            bufs[pno] = io.BytesIO()
            overlays[pno] = canvas.Canvas(bufs[pno], pagesize=(float(mb.right) - float(mb.left), float(mb.top) - float(mb.bottom)))
        return overlays[pno]

    for f, v in todo:
        page = reader.pages[f.page]
        mb = page.mediabox
        ox, oy = float(mb.left), float(mb.bottom)
        ph = template.pages[f.page].height
        ok = True
        try:
            if template.acroform and f.acro_name:
                _fill_acro(writer, f, v, warn, report)
                continue
            c = canv(f.page)
            c.translate(0, 0)
            c.saveState()
            if f.shape == "checkbox":
                if str(v).strip().lower() in _YES or v is True:
                    _tick(c, f.options[0].box if f.options else f.bbox, ph, 0, 0)
                else:
                    report["skipped"].append({"field": f.call_name, "label": f.label, "why": "answer was no"}); c.restoreState(); continue
            elif f.shape == "choice":
                i = _match_option(str(v), f)
                if i is None:
                    warn.append(f"'{f.label}': the answer '{v}' matches none of {[o.label for o in f.options]}; left blank")
                    report["skipped"].append({"field": f.call_name, "label": f.label, "why": "no matching option"}); c.restoreState(); continue
                _tick(c, f.options[i].box, ph, 0, 0)
            else:
                text = _text_for(f, v, warn)
                if f.shape == "comb":
                    ok = _draw_comb(c, text, f, ph, 0, 0)
                elif f.shape == "multiline":
                    ok = _draw_multiline(c, text, f, ph, 0, 0)
                else:
                    ok = _draw_text(c, text, f.bbox, ph, 0, 0, f.source in ("underline", "dots"))
            c.restoreState()
            if not ok:
                warn.append(f"'{f.label}': the answer is longer than the space and was cut/shrunk")
            report["filled"].append({"field": f.call_name, "label": f.label, "value": str(v), "fits": ok})
        except Exception as exc:                                       # noqa: BLE001 - one bad field must not lose the rest
            report["skipped"].append({"field": f.call_name, "label": f.label, "why": f"{type(exc).__name__}: {exc}"})
    for pno, c in overlays.items():
        c.showPage(); c.save()
        bufs[pno].seek(0)
        ov = PdfReader(bufs[pno]).pages[0]
        mb = writer.pages[pno].mediabox
        ov.mediabox = type(mb)([0, 0, float(mb.right) - float(mb.left), float(mb.top) - float(mb.bottom)])
        from pypdf import Transformation
        writer.pages[pno].merge_transformed_page(ov, Transformation().translate(float(mb.left), float(mb.bottom)))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue(), report


def _fill_acro(writer, f: TField, v: Any, warn: List[str], report: Dict[str, Any]) -> None:
    from pypdf.generic import NameObject, BooleanObject

    page = writer.pages[f.page]
    if f.shape == "checkbox":
        on = "/Yes"
        for a in page.get("/Annots") or []:
            ao = a.get_object()
            if ao.get("/T") == f.acro_name or (ao.get("/Parent") and ao["/Parent"].get_object().get("/T") == f.acro_name):
                on = next((k for k in (ao.get("/AP", {}).get("/N", {}) or {}) if k != "/Off"), "/Yes")
        val = on if (str(v).strip().lower() in _YES or v is True) else "/Off"
        writer.update_page_form_field_values(page, {f.acro_name: val}, auto_regenerate=False)
    elif f.shape == "choice":
        opts = [o.label for o in f.options]
        i = _match_option(str(v), f)
        if i is None:
            warn.append(f"'{f.label}': the answer '{v}' matches none of {opts}"); return
        writer.update_page_form_field_values(page, {f.acro_name: opts[i]}, auto_regenerate=False)
    else:
        text = _text_for(f, v, warn)
        if f.cells:
            text = re.sub(r"[\s/-]", "", text)[:len(f.cells)]
        writer.update_page_form_field_values(page, {f.acro_name: text}, auto_regenerate=False)
    writer.set_need_appearances_writer(True)
    report["filled"].append({"field": f.call_name, "label": f.label, "value": str(v), "fits": True})
