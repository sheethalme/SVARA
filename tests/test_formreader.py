"""Form reader: detection on generated sample PDFs, fill alignment (checked with an independent renderer), store, publish."""
import ast
import glob
import io
import os

import pytest

pytest.importorskip("pdfplumber"); pytest.importorskip("pypdf"); pytest.importorskip("reportlab"); pytest.importorskip("pypdfium2")
import pypdfium2 as pdfium
from PIL import ImageChops

from formreader import samples
from formreader.analyze import analyze_pdf
from formreader.fill import fill_pdf, format_date
from formreader.model import FormTemplate, TemplateError
from formreader.publish import answers_from_record, publish_template
from formreader.store import TemplateStore


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    return samples.make_all(str(tmp_path_factory.mktemp("forms")))


def by_name(t):
    return {f.call_name: f for f in t.fields if not f.ignored}


def test_college_flat(pdfs):
    t = analyze_pdf(pdfs["college_flat"])
    f = by_name(t)
    for n in ("name", "fathers_name", "date_of_birth", "age", "gender", "blood_group", "phone", "email", "address", "pincode",
              "aadhaar", "category", "course_applied_for"):
        assert n in f, n
    assert f["date_of_birth"].date_format == "DD/MM/YYYY"
    assert [o.label for o in f["gender"].options] == ["Male", "Female", "Other"]
    assert [o.label for o in f["category"].options] == ["General", "OBC", "SC", "ST"]
    assert len(f["pincode"].cells) == 6 and len(f["aadhaar"].cells) == 12
    assert f["address"].shape == "multiline" and len(f["address"].line_boxes) == 3
    ign = {x.label for x in t.fields if x.ignored}
    assert {"Application No", "Receipt No", "Signature of Applicant"} <= ign
    assert "date" in f                      # the date next to the signature is the applicant's


def test_realistic_college(pdfs):
    t = analyze_pdf(pdfs["realistic_college"])
    f = by_name(t)
    for n in ("name", "fathers_name", "mothers_name", "date_of_birth", "age", "gender", "nationality", "phone", "email",
              "permanent_address", "pincode", "aadhaar", "category", "place", "date"):
        assert n in f, n
    assert [o.label for o in f["gender"].options] == ["Male", "Female", "Other"]
    assert f["permanent_address"].shape == "multiline"
    assert {x.label for x in t.fields if x.ignored} >= {"Application No", "Signature of Candidate"}


def test_other_layouts(pdfs):
    assert {"name", "age", "gender", "phone", "blood_group", "address"} <= set(by_name(analyze_pdf(pdfs["medical_table"])))
    assert by_name(analyze_pdf(pdfs["medical_table"]))["address"].shape == "multiline"
    assert {"name", "roll_number", "department", "phone", "email"} == set(by_name(analyze_pdf(pdfs["dotted_leaders"])))
    assert {"name", "date_of_birth", "phone", "annual_family_income"} == set(by_name(analyze_pdf(pdfs["label_above_box"])))
    t = analyze_pdf(pdfs["fillable"])
    assert t.acroform and {"name", "age", "phone", "address"} <= set(by_name(t))
    t = analyze_pdf(pdfs["two_page_offset"])
    assert [(f.page, f.call_name) for f in t.fields] == [(0, "name"), (0, "phone"), (1, "place_of_birth"), (1, "nationality")]


VALUES = {"name": "Anita Rao", "fathers_name": "Suresh Rao", "date_of_birth": "2001-03-05", "age": "25", "gender": "Female",
          "blood_group": "B+", "phone": "9876543210", "email": "anita@example.com", "address": "12 MG Road Indiranagar Bengaluru",
          "pincode": "560038", "aadhaar": "234567890123", "category": "OBC", "course_applied_for": "B.Tech", "date": "2026-11-23",
          "roll_number": "21CS045", "department": "Physics", "annual_family_income": "250000", "place_of_birth": "Mysuru",
          "nationality": "Indian", "fathers_name2": "x", "mothers_name": "Latha Rao", "permanent_address": "12 MG Road Bengaluru", "chief_complaint": "Fever for two days", "full_name": "Anita Rao"}


def _render(pdf: bytes, page: int):
    doc = pdfium.PdfDocument(pdf)
    try:
        return doc[page].render(scale=2).to_pil().convert("L")
    finally:
        doc.close()


@pytest.mark.parametrize("name", ["college_flat", "medical_table", "dotted_leaders", "label_above_box", "two_page_offset", "realistic_college"])
def test_fill_lands_in_the_boxes(pdfs, name):
    """Independent check: pixels that changed between blank and filled render must lie inside the fields' boxes, and every
    filled field must have changed pixels."""
    raw = open(pdfs[name], "rb").read()
    t = analyze_pdf(raw)
    out, rep = fill_pdf(t, raw, VALUES)
    assert not rep["warnings"], rep["warnings"]
    assert rep["filled"]
    s = 2
    for p in range(len(t.pages)):
        diff = ImageChops.difference(_render(raw, p), _render(out, p)).point(lambda v: 255 if v > 40 else 0)
        allowed = [b for f in t.fields if f.page == p and not f.ignored for b in ([f.bbox] + [o.box for o in f.options] + f.cells + f.line_boxes)]
        W_, H_ = diff.size
        px = diff.load()
        for y in range(H_):
            for x in range(W_):
                if px[x, y] and not any(b[0] * s - 3 <= x <= b[2] * s + 3 and b[1] * s - 4 <= y <= b[3] * s + 5 for b in allowed):
                    pytest.fail(f"{name}: changed pixel ({x / s:.0f},{y / s:.0f}) is outside every field box")
        filled_labels = {r["field"] for r in rep["filled"]}
        for f in t.fields:
            if f.page == p and f.call_name in filled_labels:
                boxes = [o.box for o in f.options] if f.shape == "choice" else [f.bbox]
                assert any(diff.crop((int(b[0] * s - 3), int(b[1] * s - 4), int(b[2] * s + 3), int(b[3] * s + 5))).getbbox() for b in boxes), \
                    f"{name}: nothing was written for {f.call_name}"


def test_fillable_pdf(pdfs):
    from pypdf import PdfReader
    raw = open(pdfs["fillable"], "rb").read()
    t = analyze_pdf(raw)
    out, rep = fill_pdf(t, raw, {"name": "Anita Rao", "age": "25", "i_agree_to_the_terms": "yes"})
    fields = PdfReader(io.BytesIO(out)).get_fields()
    assert fields["full_name"]["/V"] == "Anita Rao" and fields["age"]["/V"] == "25"
    assert str(fields["agree"]["/V"]) != "/Off"


def test_choice_without_match_is_reported(pdfs):
    raw = open(pdfs["college_flat"], "rb").read()
    t = analyze_pdf(raw)
    _, rep = fill_pdf(t, raw, {"category": "Martian", "name": "Zoë ✓"})
    assert any("matches none" in w for w in rep["warnings"]) and any("left out" in w for w in rep["warnings"])


def test_long_text_is_reported(pdfs):
    raw = open(pdfs["college_flat"], "rb").read()
    _, rep = fill_pdf(analyze_pdf(raw), raw, {"pincode": "56003812345", "age": "x" * 200})
    assert any("cut" in w or "longer" in w for w in rep["warnings"])


def test_date_format():
    assert format_date("2001-03-05", None) == "05/03/2001"
    assert format_date("2001-03-05", "DD-MM-YYYY") == "05-03-2001"
    assert format_date("2001-03-05", "MM/DD/YYYY") == "03/05/2001"
    assert format_date("2001-03-05", "DDMMYYYY") == "05032001"
    assert format_date("5 March", None) == "5 March"


def test_scan_warning(tmp_path):
    from PIL import Image
    p = tmp_path / "scan.pdf"
    Image.new("RGB", (400, 600), "white").save(p)
    t = analyze_pdf(str(p))
    assert any("SCAN" in w for w in t.warnings) and not t.fields


def test_store_and_validation(pdfs, tmp_path):
    raw = open(pdfs["dotted_leaders"], "rb").read()
    t = analyze_pdf(raw)
    st = TemplateStore(str(tmp_path))
    st.save(t, raw)
    assert st.list() == [(t.id, t.title)] and st.load(t.id).to_dict() == t.to_dict() and st.pdf_bytes(t.id) == raw
    with pytest.raises(TemplateError):
        st.load("../x")
    t.fields[0].call_name = "Bad Name"
    with pytest.raises(TemplateError):
        t.validate()


def test_publish_and_answers(pdfs, tmp_path):
    from callpage.forms import FormStore
    t = analyze_pdf(pdfs["college_flat"])
    spec = publish_template(t, FormStore(str(tmp_path)))
    names = [f.name for f in spec.fields]
    assert "name" in names and "aadhaar" in names and "application_no" not in names
    q = spec.field_by_name("gender").question_for("en")
    assert "Male" in q and "Female" in q
    rec = {"fields": [{"name": "name", "status": "confirmed", "value": "Anita"}, {"name": "age", "status": "skipped", "value": None},
                      {"name": "phone", "status": "confirmed", "value": "9876543210"}],
           "corrections": [{"field": "name", "value": "Anita Rao"}]}
    assert answers_from_record(rec) == {"name": "Anita Rao", "phone": "9876543210"}
    rec["corrections"] = [{"field": "name", "old": "Anita", "new": "Anita Corrected"}]
    assert answers_from_record(rec) == {"name": "Anita Corrected", "phone": "9876543210"}


def test_no_network_imports():
    bad = {"requests", "urllib", "urllib3", "http", "socket", "httpx", "aiohttp", "groq", "openai", "ftplib", "smtplib"}
    root = os.path.join(os.path.dirname(__file__), "..", "formreader")
    for path in glob.glob(os.path.join(root, "*.py")):
        for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for m in mods:
                assert m.split(".")[0] not in bad, f"{os.path.basename(path)} imports {m}"
