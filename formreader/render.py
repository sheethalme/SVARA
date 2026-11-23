"""render.py -- page images for the review screen (numbered boxes over the form) and for previewing the filled PDF."""

from __future__ import annotations

import io
from typing import Optional

from formreader.model import FormTemplate

_COLORS = {"ok": (0, 140, 60), "low": (230, 120, 0), "ignored": (130, 130, 130)}


def render_page(pdf_bytes: bytes, page: int = 0, scale: float = 1.5) -> bytes:
    """PNG of one page of any PDF (uses pypdfium2; no network)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        img = doc[page].render(scale=scale).to_pil().convert("RGB")
        buf = io.BytesIO(); img.save(buf, "PNG")
        return buf.getvalue()
    finally:
        doc.close()


def render_overlay(pdf_bytes: bytes, template: FormTemplate, page: int = 0, scale: float = 1.5) -> bytes:
    """The page with every detected field boxed and numbered (green = sure, orange = check, grey = ignored)."""
    from PIL import Image, ImageDraw
    img = Image.open(io.BytesIO(render_page(pdf_bytes, page, scale))).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    # pypdfium2 draws the page's visible box at (0,0): shift by the template page box is not needed because boxes are
    # stored relative to that visible box (pdfplumber convention).
    for i, f in enumerate(template.fields, 1):
        if f.page != page:
            continue
        col = _COLORS["ignored"] if f.ignored else (_COLORS["low"] if f.confidence < 0.6 else _COLORS["ok"])
        boxes = [f.bbox] + [o.box for o in f.options]
        for x0, t, x1, b in boxes:
            d.rectangle([x0 * scale, t * scale, x1 * scale, b * scale], outline=col + (255,), fill=col + (40,), width=2)
        x0, t = f.bbox[0], f.bbox[1]
        tag = str(i)
        d.rectangle([x0 * scale - 1, t * scale - 14, x0 * scale + 8 * len(tag) + 5, t * scale], fill=col + (255,))
        d.text((x0 * scale + 2, t * scale - 13), tag, fill=(255, 255, 255, 255))
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()
