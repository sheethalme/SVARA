"""
analyze.py -- reads a DIGITAL PDF form and finds the places to fill in (locally: no cloud, no AI)
===================================================================================================

    template = analyze_pdf("college_form.pdf")          # -> FormTemplate (fields with page, box, label, meaning)

How it works
------------
1. If the PDF has real fillable fields (AcroForm widgets) they are read directly: name, type, position, tooltip.
2. Otherwise ("flat" forms) the page is read as text + graphics:
   * words with positions (underscore runs "______" and dot leaders "........" are blanks, not words);
   * horizontal lines  = underline blanks;   empty rectangles = boxes;   rows of equal touching squares = one-letter-per-box
     fields ("comb");   small squares followed by text = tick boxes;   ruled tables = label | empty cell pairs.
3. Every blank gets a LABEL: the words to its left on the same line, else the words written on the line itself
   ("Name: ______" where the line also runs under the label), else the closest line of text above it.
4. Blanks under a label with no label of their own continue it (a 3-line address).
5. Tick boxes on one row become one choice field ("Gender: [ ] Male [ ] Female").
6. Fields below "FOR OFFICE USE ONLY", signatures and photo areas are marked ``ignored`` (not asked).
7. The label is turned into a meaning (name, age, phone, date of birth ...) with the rules of ``semantics``.

Limits (phase 1): digital text PDFs only (scans / photos need OCR), English, single-page-rotation 0, one AcroForm OR
flat reading per file.  Detection is a best effort: the review screen exists so a person can fix what it gets wrong.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pdfplumber

from formreader import semantics
from formreader.model import Box, FormTemplate, Option, PageInfo, TField

PdfSource = Union[str, bytes, os.PathLike]
_RUN = re.compile(r"[_…]{3,}|\.{4,}")
_TICK_GLYPHS = set("☐□▢❏❑❒◻⬜")           # ☐ □ ▢ ❏ ❑ ❒ ◻ ⬜
_OFFICE = re.compile(r"(for\s+)?(office|official)\s+use(\s+only)?|to\s+be\s+filled\s+(in\s+)?by\s+(the\s+)?(office|official)", re.I)


# ------------------------------------------------------------------------------------------------ elements
@dataclass
class W:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float = 10.0
    used: bool = False


@dataclass
class Cand:
    kind: str                       # line | dots | box | comb | tick
    x0: float
    x1: float
    top: float
    bottom: float
    cells: List[Box] = field(default_factory=list)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ------------------------------------------------------------------------------------------------ page reading
def _read_words(page) -> Tuple[List[W], List[Cand]]:
    """Words plus the blanks that are written as text (underscore runs, dot leaders)."""
    words: List[W] = []
    blanks: List[Cand] = []
    chars = page.chars
    for w in page.extract_words(x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False, extra_attrs=["size"]):
        text = w["text"]
        runs = list(_RUN.finditer(text))
        if not runs:
            words.append(W(text, w["x0"], w["x1"], w["top"], w["bottom"], w.get("size", 10.0)))
            continue
        inside = sorted((c for c in chars if w["x0"] - 0.6 <= c["x0"] and c["x1"] <= w["x1"] + 0.6
                         and w["top"] - 1 <= c["top"] and c["bottom"] <= w["bottom"] + 1), key=lambda c: c["x0"])
        if len(inside) != len(text):                                    # ligatures etc.: fall back to a proportional split
            width = (w["x1"] - w["x0"]) / max(1, len(text))
            pos = [(w["x0"] + i * width, w["x0"] + (i + 1) * width) for i in range(len(text))]
        else:
            pos = [(c["x0"], c["x1"]) for c in inside]
        last = 0
        for m in runs:
            if m.start() > last:
                words.append(W(text[last:m.start()], pos[last][0], pos[m.start() - 1][1], w["top"], w["bottom"], w.get("size", 10.0)))
            blanks.append(Cand("dots", pos[m.start()][0], pos[m.end() - 1][1], w["top"], w["bottom"]))
            last = m.end()
        if last < len(text):
            words.append(W(text[last:], pos[last][0], pos[-1][1], w["top"], w["bottom"], w.get("size", 10.0)))
    for c in chars:                                                     # glyph check boxes
        if c["text"] in _TICK_GLYPHS:
            blanks.append(Cand("tick", c["x0"], c["x1"], c["top"], c["bottom"]))
    words = [w for w in words if w.text.strip() and not (w.text in _TICK_GLYPHS)]
    return words, blanks


def _merge_hlines(segs: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Dashed lines come as many short pieces: join pieces on the same y that nearly touch."""
    segs = sorted(segs, key=lambda s: (round(s[2] / 1.2), s[0]))
    out: List[List[float]] = []
    for x0, x1, y in segs:
        if out and abs(out[-1][2] - y) <= 1.2 and x0 - out[-1][1] <= 4.5:
            out[-1][1] = max(out[-1][1], x1)
        else:
            out.append([x0, x1, y])
    return [(a, b, c) for a, b, c in out]


def _is_white(color: Any) -> bool:
    if color is None:
        return True
    try:
        vals = list(color) if isinstance(color, (list, tuple)) else [color]
        return all(isinstance(v, (int, float)) and v >= 0.93 for v in vals) if len(vals) in (1, 3) else False
    except TypeError:
        return False


def _comb_groups(rects: List[Dict[str, Any]]) -> Tuple[List[Cand], set]:
    """Rows of equal squares that touch are one-letter-per-box fields."""
    cand = [r for r in rects if 8 <= r["width"] <= 40 and 8 <= r["height"] <= 40]
    cand.sort(key=lambda r: (round(r["top"] / 2), r["x0"]))
    chains: List[List[Dict[str, Any]]] = []
    for r in cand:
        if chains:
            p = chains[-1][-1]
            if abs(p["top"] - r["top"]) <= 1.6 and abs(p["bottom"] - r["bottom"]) <= 1.6 and abs(p["width"] - r["width"]) <= 1.6 \
                    and -1.5 <= r["x0"] - p["x1"] <= 3.8:
                chains[-1].append(r); continue
        chains.append([r])
    # "DD MM YYYY": short chains (2, 2, 4 cells) with a small gap between them are one date field
    merged: List[List[Dict[str, Any]]] = []
    for ch in chains:
        if merged and len(ch) in (2, 4) and len(merged[-1]) in (2, 4, 6) and abs(merged[-1][-1]["top"] - ch[0]["top"]) <= 1.6 \
                and 0 <= ch[0]["x0"] - merged[-1][-1]["x1"] <= 20 and abs(merged[-1][-1]["width"] - ch[0]["width"]) <= 1.6 \
                and len(merged[-1]) + len(ch) <= 8 and len(merged[-1]) >= 2:
            merged[-1] = merged[-1] + ch
        else:
            merged.append(ch)
    out: List[Cand] = []
    used = set()
    for ch in merged:
        if len(ch) >= 4:
            out.append(Cand("comb", ch[0]["x0"], ch[-1]["x1"], min(r["top"] for r in ch), max(r["bottom"] for r in ch),
                            [(r["x0"], r["top"], r["x1"], r["bottom"]) for r in ch]))
            used.update(id(r) for r in ch)
    return out, used


# ------------------------------------------------------------------------------------------------ labels
def _row_band(c: Cand) -> Tuple[float, float]:
    if c.kind == "line":
        return c.top - 12.5, c.top + 1.5
    return c.top - 1.0, c.bottom + 1.0


def _left_label(c: Cand, band: Tuple[float, float], words: List[W], obstacles: List[Cand], limit_gap: float = 150.0) -> List[W]:
    """Words to the left of ``c`` on its row, back to the previous blank / large gap."""
    mid = (band[0] + band[1]) / 2
    cands = [w for w in words if not w.used and w.x1 <= c.x0 + 2.0 and _overlap(w.top, w.bottom, band[0], band[1]) >= 0.45 * (w.bottom - w.top)
             and w.top - 2 <= mid <= w.bottom + 6]
    left_limit = max([o.x1 for o in obstacles if o is not c and o.x1 <= c.x0 + 1.0 and _overlap(o.top, o.bottom, band[0], band[1]) > 0],
                     default=-1e9)
    cands = [w for w in cands if w.x0 >= left_limit - 1.0]
    cands.sort(key=lambda w: -w.x1)
    run: List[W] = []
    edge = c.x0
    for w in cands:
        gap = edge - w.x1
        if (not run and gap > limit_gap) or (run and gap > 14.0):
            break
        run.append(w); edge = w.x0
    run.sort(key=lambda w: w.x0)
    return run


def _above_label(c: Cand, words: List[W]) -> List[W]:
    """The closest short line of text above ``c`` that sits over it."""
    near = [w for w in words if not w.used and w.bottom <= c.top + 2.0 and w.bottom >= c.top - 28.0
            and _overlap(w.x0, w.x1, c.x0 - 2, c.x1 + 2) > 0]
    if not near:
        return []
    ref = max(near, key=lambda w: w.bottom)
    line = [w for w in near if abs(w.bottom - ref.bottom) <= 3.0]
    full = [w for w in words if not w.used and abs(w.bottom - ref.bottom) <= 3.0 and w.x1 >= min(x.x0 for x in line) - 14
            and w.x0 <= max(x.x1 for x in line) + 14]
    if c.x1 - c.x0 < 200:                                               # a narrow box: only the words above it
        line = [w for w in line if _overlap(w.x0, w.x1, c.x0 - 6, c.x1 + 6) > 0]
    line.sort(key=lambda w: w.x0)
    text = " ".join(w.text for w in line)
    if len(text) > 70 or len(line) > 10 or not line:
        return []
    return line


def _inline_label(c: Cand, words: List[W]) -> Tuple[List[W], float]:
    """"Name: ______" where the line also runs under the label: words ON the line, starting at its left end."""
    on = sorted((w for w in words if not w.used and c.x0 - 1 <= w.x0 and w.x1 <= c.x1 + 1 and c.top - 12.5 <= w.bottom <= c.top + 1.5), key=lambda w: w.x0)
    if not on or on[0].x0 - c.x0 > 6:
        return [], c.x0
    run = [on[0]]
    for w in on[1:]:
        if w.x0 - run[-1].x1 <= 12:
            run.append(w)
        else:
            break
    covered = sum(w.x1 - w.x0 for w in on) / max(1.0, c.x1 - c.x0)
    if covered >= 0.6:
        return [], -1.0                                                  # underlined text, not a blank
    return run, run[-1].x1 + 4.0


# ------------------------------------------------------------------------------------------------ one page
def _page_fields(page, pno: int, notes: List[str]) -> List[TField]:
    words, text_blanks = _read_words(page)
    rects = [r for r in page.rects if r.get("stroke") and not (r.get("fill") and not _is_white(r.get("non_stroking_color")))
             and r["width"] > 1.6 and r["height"] > 1.6 and not (r["width"] > 0.6 * page.width and r["height"] > 0.5 * page.height)]
    thin = [r for r in page.rects if r["height"] <= 1.6 and r["width"] >= 20]
    combs, comb_ids = _comb_groups(rects)
    rects = [r for r in rects if id(r) not in comb_ids]

    # ---- tables (label | empty cell)
    table_fields: List[TField] = []
    table_boxes: List[Box] = []
    try:
        tables = page.find_tables()
    except Exception:                                                   # noqa: BLE001 - odd graphics must not stop the reading
        tables = []
    for t in tables:
        cells = [c for c in t.cells if c]
        cs = [Cand("box", c[0], c[2], c[1], c[3]) for c in cells]
        if len(cells) < 2:
            continue
        tb = t.bbox
        if (tb[2] - tb[0]) > 0.6 * page.width and (tb[3] - tb[1]) > 0.5 * page.height and len(cells) < 6:
            continue                                                     # a page border, not a table
        if any(abs(c.x0 - cb.x0) < 1 and abs(c.top - cb.top) < 1 for c in cs for cb in combs):
            continue                                                     # that "table" is a comb field
        with_text = sum(1 for c in cells if any(c[0] - 1 <= w.x0 and w.x1 <= c[2] + 1 and c[1] - 1 <= w.top and w.bottom <= c[3] + 1 for w in words))
        tf = _table_fields(t, words, pno)
        if len(tf) < 2 or with_text < 0.34 * len(cells):
            continue                                                     # ruled lines that merely look like a grid
        table_boxes.append(t.bbox)
        table_fields += tf
    in_table = lambda c: any(c.x0 >= b[0] - 1 and c.x1 <= b[2] + 1 and c.top >= b[1] - 1 and c.bottom <= b[3] + 1 for b in table_boxes)

    # ---- candidates
    cands: List[Cand] = []
    ticks: List[Cand] = [b for b in text_blanks if b.kind == "tick"]
    cands += [b for b in text_blanks if b.kind == "dots"]
    for r in rects:
        c = Cand("box", r["x0"], r["x1"], r["top"], r["bottom"])
        if in_table(c):
            continue
        if 6 <= r["width"] <= 16 and 6 <= r["height"] <= 16 and abs(r["width"] - r["height"]) <= 2.5:
            ticks.append(Cand("tick", r["x0"], r["x1"], r["top"], r["bottom"]))
        elif r["width"] >= 25 and r["height"] >= 9 and not any(w.x0 >= r["x0"] and w.x1 <= r["x1"] and w.top >= r["top"] - 1 and w.bottom <= r["bottom"] + 1 for w in words):
            cands.append(c)
    cands += [c for c in combs if not in_table(c)]
    ticks = [t for t in ticks if not in_table(t)]
    tick_fields = _tick_fields(ticks, words, cands + ticks, pno)         # first, so option words are not taken as labels
    segs = [(l["x0"], l["x1"], (l["top"] + l["bottom"]) / 2) for l in page.lines if abs(l["bottom"] - l["top"]) <= 1.5 and l["x1"] - l["x0"] >= 20]
    segs += [(r["x0"], r["x1"], (r["top"] + r["bottom"]) / 2) for r in thin]
    for x0, x1, y in _merge_hlines(segs):
        c = Cand("line", x0, x1, y, y)
        if not in_table(c):
            cands.append(c)
    obstacles = cands + ticks
    cands.sort(key=lambda c: (round(c.top / 5), c.x0))

    fields: List[TField] = list(table_fields)
    for f in table_fields:                                               # words inside table cells are used up
        for w in words:
            if f.bbox[0] - 200 <= w.x0 and w.x1 <= f.bbox[2] and f.bbox[1] - 3 <= w.top and w.bottom <= f.bbox[3] + 3:
                pass
    for b in table_boxes:
        for w in words:
            if b[0] - 1 <= w.x0 and w.x1 <= b[2] + 1 and b[1] - 1 <= w.top and w.bottom <= b[3] + 1:
                w.used = True

    last_line: Optional[TField] = None
    for c in cands:
        label_words: List[W] = []
        how = ""
        x0 = c.x0
        if c.kind == "line":
            inline, newx0 = _inline_label(c, words)
            if newx0 < 0:
                continue                                                 # underlined heading / text
            if inline:
                label_words, how, x0 = inline, "inline", newx0
        band = _row_band(c)
        if not label_words:
            label_words = _left_label(c, band, words, obstacles)
            how = "left" if label_words else ""
        if not label_words and c.kind in ("line", "box", "dots", "comb"):
            label_words = _above_label(c, words)
            how = "above" if label_words else ""
        bbox = (x0, c.top - (11.0 if c.kind == "line" else 0.0) if c.kind == "line" else c.top, c.x1, c.bottom if c.kind != "line" else c.top + 1.0)
        shape = {"line": "text", "dots": "text", "box": "text", "comb": "comb"}.get(c.kind, "text")
        if c.kind == "line":
            bbox = (x0, c.top - 12.0, c.x1, c.top)
        elif c.kind == "dots":
            bbox = (x0, c.top - 1.0, c.x1, c.bottom + 1.0)
        elif c.kind == "box" and (c.bottom - c.top) >= 40:
            shape = "multiline"
        if not label_words:
            if c.kind in ("line", "dots") and last_line is not None and last_line.page == pno and 0 < c.top - last_line.bbox[3] <= 32 \
                    and -2 <= last_line.bbox[0] - c.x0 <= 140 and last_line.source in ("underline", "dots"):
                lb = (c.x0, c.top - 12.0, c.x1, c.top) if c.kind == "line" else bbox
                if not last_line.line_boxes:
                    last_line.line_boxes = [last_line.bbox]
                last_line.line_boxes.append(lb)
                last_line.shape = "multiline"
                last_line.bbox = (min(last_line.bbox[0], lb[0]), last_line.bbox[1], max(last_line.bbox[2], lb[2]), lb[3])
                continue
            label, req, hints = "", True, {}
            conf, note = 0.3, ["no label found next to this blank"]
        else:
            for w in label_words:
                w.used = True
            label, req, hints = semantics.clean_label(" ".join(w.text for w in label_words))
            conf, note = {"left": 0.9, "inline": 0.85, "above": 0.75}.get(how, 0.5), []
            if not label:
                conf, note = 0.3, ["the label could not be read"]
        f = TField(id="", label=label, page=pno, bbox=bbox, shape=shape, required=req,
                   source={"line": "underline", "dots": "dots", "box": "box", "comb": "comb"}[c.kind], confidence=conf, notes=note,
                   cells=list(c.cells), date_format=hints.get("date_format"))
        fields.append(f)
        last_line = f if c.kind in ("line", "dots") else None

    fields += tick_fields

    # ---- sections that are not for the user
    heading = None
    for ln in _text_lines(words):
        if _OFFICE.search(ln[0]):
            heading = ln[1]
            break
    if heading is not None:
        for f in sorted(fields, key=lambda f: (round(f.bbox[1] / 6), f.bbox[0])):
            if f.bbox[1] < heading - 2:
                continue
            if semantics.infer_kind(f.label, f.shape) == "signature":
                break                                                    # the office block ends where the applicant signs
            f.ignored = True
            f.notes.append("below 'FOR OFFICE USE ONLY'")
    return fields


def _text_lines(words: List[W]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for w in sorted(words, key=lambda w: (round(w.top / 3), w.x0)):
        if out and abs(out[-1][1] - w.top) <= 3:
            out[-1] = (out[-1][0] + " " + w.text, out[-1][1])
        else:
            out.append((w.text, w.top))
    return out


def _table_fields(table, words: List[W], pno: int) -> List[TField]:
    out: List[TField] = []
    rows = table.rows
    header_style = False
    if len(rows) >= 2:
        first = [c for c in rows[0].cells if c]
        has = lambda c: any(c[0] - 1 <= w.x0 and w.x1 <= c[2] + 1 and c[1] - 1 <= w.top and w.bottom <= c[3] + 1 for w in words)
        header_style = len(first) >= 2 and all(has(c) for c in first) and not any(has(c) for r in rows[1:] for c in r.cells if c)
    for ri, row in enumerate(rows):
        cells = list(row.cells)
        for ci, cell in enumerate(cells):
            if not cell:
                continue
            inside = [w for w in words if cell[0] - 1 <= w.x0 and w.x1 <= cell[2] + 1 and cell[1] - 1 <= w.top and w.bottom <= cell[3] + 1]
            if inside:
                continue
            label_words: List[W] = []
            how = ""
            left = next((cells[j] for j in range(ci - 1, -1, -1) if cells[j]), None)
            if left:
                label_words = [w for w in words if left[0] - 1 <= w.x0 and w.x1 <= left[2] + 1 and left[1] - 1 <= w.top and w.bottom <= left[3] + 1]
                how = "table"
            if not label_words and header_style and ri == 1:
                hdr = rows[0].cells[ci] if ci < len(rows[0].cells) else None
                if hdr:
                    label_words = [w for w in words if hdr[0] - 1 <= w.x0 and w.x1 <= hdr[2] + 1 and hdr[1] - 1 <= w.top and w.bottom <= hdr[3] + 1]
                    how = "table header"
            if not label_words:
                continue
            if header_style and ri > 1:
                continue                                                 # only the first empty row of a header-style table
            label, req, hints = semantics.clean_label(" ".join(w.text for w in sorted(label_words, key=lambda w: (round(w.top / 3), w.x0))))
            if not label:
                continue
            h = cell[3] - cell[1]
            f = TField(id="", label=label, page=pno, bbox=(cell[0] + 1.5, cell[1] + 1.5, cell[2] - 1.5, cell[3] - 1.5),
                       shape="multiline" if h >= 45 else "text", required=req, source="table", confidence=0.85,
                       date_format=hints.get("date_format"))
            if header_style:
                f.notes.append("a table with several rows: only the first row is filled")
            out.append(f)
    return out


def _tick_fields(ticks: List[Cand], words: List[W], obstacles: List[Cand], pno: int) -> List[TField]:
    """Tick boxes -> options; options on one row (or one column) with their label become one choice field."""
    items: List[Tuple[Cand, str]] = []
    for t in sorted(ticks, key=lambda t: (round(t.top / 4), t.x0)):
        right = [w for w in words if not w.used and w.x0 >= t.x1 - 1 and _overlap(w.top, w.bottom, t.top - 2, t.bottom + 2) > 0.4 * (w.bottom - w.top)]
        nxt = min([o.x0 for o in obstacles if o is not t and o.x0 > t.x1 + 1 and _overlap(o.top, o.bottom, t.top - 2, t.bottom + 2) > 0], default=1e9)
        right = sorted((w for w in right if w.x1 <= nxt + 1), key=lambda w: w.x0)
        run: List[W] = []
        edge = t.x1
        for w in right:
            if w.x0 - edge > 9:
                break
            run.append(w); edge = w.x1
        run = run[:6]
        for w in run:
            w.used = True
        items.append((t, " ".join(w.text for w in run)))
    out: List[TField] = []
    rows: List[List[Tuple[Cand, str]]] = []
    for it in items:
        if rows and abs(rows[-1][0][0].top - it[0].top) <= 4:
            rows[-1].append(it)
        else:
            rows.append([it])
    groups: List[List[Tuple[Cand, str]]] = []
    for row in rows:                                                      # a vertical list: single boxes stacked at one x
        if len(row) == 1 and groups and len(groups[-1]) >= 1 and abs(groups[-1][-1][0].x0 - row[0][0].x0) <= 4 \
                and 0 < row[0][0].top - groups[-1][-1][0].bottom <= 24 and all(True for _ in [0]) \
                and not _left_label(row[0][0], (row[0][0].top - 1, row[0][0].bottom + 1), words, obstacles):
            groups[-1].append(row[0])
        else:
            groups.append(list(row))
    for g in groups:
        first = g[0][0]
        lab = _left_label(first, (first.top - 1, first.bottom + 1), words, [o for o in obstacles if o is not first and o.kind != "tick"])
        how = "left"
        if not lab:
            lab = _above_label(first, words); how = "above"
        for w in lab:
            w.used = True
        label_text = " ".join(w.text for w in lab)
        label, req, hints = semantics.clean_label(label_text)
        boxes = [Option(t[1] or f"option {i + 1}", (t[0].x0, t[0].top, t[0].x1, t[0].bottom)) for i, t in enumerate(g)]
        x0 = min(b.box[0] for b in boxes); x1 = max(b.box[2] for b in boxes); top = min(b.box[1] for b in boxes); bot = max(b.box[3] for b in boxes)
        if len(g) == 1 and not label:
            out.append(TField(id="", label=semantics.clean_label(g[0][1])[0] or "Tick box", page=pno, bbox=(x0, top, x1, bot),
                              shape="checkbox", source="tick", confidence=0.7, options=boxes))
        elif len(g) == 1:
            out.append(TField(id="", label=label, page=pno, bbox=(x0, top, x1, bot), shape="checkbox", source="tick",
                              confidence=0.7, options=boxes))
        else:
            out.append(TField(id="", label=label or "Choose one", page=pno, bbox=(x0, top, x1, bot), shape="choice", source="tick",
                              confidence=0.85 if label else 0.4, required=req, options=boxes,
                              notes=[] if label else ["no label found for this choice"]))
    return out


# ------------------------------------------------------------------------------------------------ fillable PDFs
def _acro_fields(reader, plumber_pdf) -> List[TField]:
    from pypdf.generic import ArrayObject, NameObject

    out: List[TField] = []
    radios: Dict[str, TField] = {}
    for pno, page in enumerate(reader.pages):
        height = plumber_pdf.pages[pno].height
        words, _ = _read_words(plumber_pdf.pages[pno])
        for annot in page.get("/Annots") or []:
            a = annot.get_object()
            if a.get("/Subtype") != "/Widget":
                continue
            node, name, ft, ff, tu, maxlen, opt = a, [], None, 0, None, None, None
            while node is not None:
                n = node.get("/T")
                if n is not None:
                    name.insert(0, str(n))
                ft = ft or node.get("/FT")
                ff = ff or int(node.get("/Ff", 0) or 0)
                tu = tu or node.get("/TU")
                maxlen = maxlen or node.get("/MaxLen")
                opt = opt or node.get("/Opt")
                p = node.get("/Parent")
                node = p.get_object() if p is not None else None
            full = ".".join(name)
            x0, y0, x1, y1 = [float(v) for v in a["/Rect"]]
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            bbox = (x0, height - y1, x1, height - y0)
            probe = Cand("box", bbox[0], bbox[2], bbox[1], bbox[3])
            lab = [w for w in _left_label(probe, (bbox[1] - 1, bbox[3] + 1), words, [])] or _above_label(probe, words)
            label_raw = str(tu) if tu else " ".join(w.text for w in lab)
            label, req, hints = semantics.clean_label(label_raw)
            label = label or full
            if ft == "/Btn":
                on = next((k for k in (a.get("/AP", {}).get("/N", {}) or {}) if k != "/Off"), "/Yes")
                if ff & (1 << 16):
                    continue                                              # push button
                if ff & (1 << 15):                                        # radio group: one choice field per group
                    g = radios.get(full)
                    if g is None:
                        g = TField(id="", label=label, page=pno, bbox=bbox, shape="choice", source="acroform", confidence=0.8, acro_name=full)
                        radios[full] = g; out.append(g)
                    opt_lab = " ".join(w.text for w in _left_label(Cand("tick", bbox[0], bbox[2], bbox[1], bbox[3]), (bbox[1], bbox[3]), words, [])[:0]) or on.lstrip("/")
                    right = [w for w in words if w.x0 >= bbox[2] - 1 and _overlap(w.top, w.bottom, bbox[1], bbox[3]) > 0][:3]
                    g.options.append(Option(" ".join(w.text for w in right) or opt_lab, bbox))
                    g.notes.append(f"radio value {on.lstrip('/')}") if len(g.notes) < 1 else None
                    continue
                out.append(TField(id="", label=label, page=pno, bbox=bbox, shape="checkbox", source="acroform", confidence=0.9,
                                  options=[Option(label, bbox)], acro_name=full))
                continue
            if ft == "/Ch":
                opts = [str(o if not isinstance(o, ArrayObject) else o[-1]) for o in (opt or [])]
                f = TField(id="", label=label, page=pno, bbox=bbox, shape="choice" if opts else "text", source="acroform", confidence=0.85,
                           options=[Option(o, bbox) for o in opts], acro_name=full, required=req)
                out.append(f)
                continue
            if ft == "/Tx":
                shape = "multiline" if ff & 4096 else "text"
                cells: List[Box] = []
                if ff & (1 << 24) and maxlen:
                    n = int(maxlen); w_ = (x1 - x0) / n
                    cells = [(x0 + i * w_, bbox[1], x0 + (i + 1) * w_, bbox[3]) for i in range(n)]
                    shape = "comb"
                out.append(TField(id="", label=label, page=pno, bbox=bbox, shape=shape, source="acroform", confidence=0.95 if tu else 0.8,
                                  cells=cells, required=req, acro_name=full, date_format=hints.get("date_format")))
    return out


# ------------------------------------------------------------------------------------------------ public API
def _title(pdf, fallback: str) -> str:
    try:
        page = pdf.pages[0]
        words, _ = _read_words(page)
        words = [w for w in words if w.top < page.height * 0.35]
        if not words:
            return fallback
        big = max(w.size for w in words)
        lines = [w for w in words if w.size >= big - 0.6]
        lines.sort(key=lambda w: (round(w.top / 3), w.x0))
        text = " ".join(w.text for w in lines)[:90].strip()
        return text if len(text) >= 4 else fallback
    except Exception:                                                    # noqa: BLE001
        return fallback


def _sort_fields(fields: List[TField]) -> List[TField]:
    return sorted(fields, key=lambda f: (f.page, round(f.bbox[1] / 6), f.bbox[0]))


def analyze_pdf(source: PdfSource, template_id: Optional[str] = None, title: Optional[str] = None,
                filename: Optional[str] = None) -> FormTemplate:
    """Read a digital PDF form and return its template.  ``source`` is a path or the PDF bytes."""
    from pypdf import PdfReader

    data = source if isinstance(source, (bytes, bytearray)) else None
    stream: Any = io.BytesIO(bytes(data)) if data is not None else str(source)
    name = filename or (os.path.basename(str(source)) if data is None else "form.pdf")
    warnings: List[str] = []
    with pdfplumber.open(stream) as pdf:
        if not pdf.pages:
            raise ValueError("the PDF has no pages")
        pages = [PageInfo(float(p.width), float(p.height), int(p.rotation or 0)) for p in pdf.pages]
        if any(p.rotation for p in pages):
            warnings.append("Some pages are rotated; the form reader works best on upright pages. Check the preview carefully.")
        n_chars = sum(len(p.chars) for p in pdf.pages)
        n_images = sum(len(p.images) for p in pdf.pages)
        if n_chars < 20 and n_images:
            warnings.append("This PDF looks like a SCAN or PHOTO (it has pictures but almost no text). Phase 1 reads digital "
                            "PDFs only; scans need OCR, which is planned for a later phase.")
        fields: List[TField] = []
        reader = PdfReader(io.BytesIO(bytes(data)) if data is not None else str(source))
        acro = False
        try:
            acro = bool(reader.get_fields())
        except Exception:                                                # noqa: BLE001
            acro = False
        if acro:
            fields = _acro_fields(reader, pdf)
        if not fields:
            acro = False
            for pno, page in enumerate(pdf.pages):
                notes: List[str] = []
                fields += _page_fields(page, pno, notes)
        fields = _sort_fields(fields)
        for i, f in enumerate(fields, 1):
            f.id = f"f{i}"
            if not f.label:
                f.label = f"Field {i}"
            if f.shape == "choice" or f.shape == "checkbox":
                pass
            kind = semantics.infer_kind(f.label, f.shape, len(f.cells) or None)
            if kind in ("signature", "photo"):
                f.shape, f.ignored = kind, True
                f.notes.append(f"{kind} areas are not asked in the call")
            if f.date_format is None and (kind in ("date", "date_of_birth")):
                f.date_format = "DD/MM/YYYY" if not f.cells else ("DDMMYYYY" if len(f.cells) == 8 else "DDMMYY" if len(f.cells) == 6 else "DD/MM/YYYY")
        semantics.assign_names(fields)
        doc_title = title or _title(pdf, os.path.splitext(name)[0].replace("_", " ").title())
    if not fields:
        warnings.append("No fields were found. The PDF may be a scan, or use a layout this version does not understand.")
    low = [f for f in fields if f.confidence < 0.5 and not f.ignored]
    if low:
        warnings.append(f"{len(low)} field(s) have a low confidence: please check them in the review screen.")
    tid = template_id or re.sub(r"[^a-z0-9]+", "_", doc_title.lower()).strip("_")[:50].strip("_") or "form"
    t = FormTemplate(tid, doc_title, name, pages, fields, warnings, acro)
    t.validate()
    return t
