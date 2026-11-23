"""
samples.py -- generates SAMPLE digital PDF forms (several different layouts) for tests and demos
================================================================================================

Real college / hospital forms come in many layouts.  These samples imitate the common ones so the form reader can be
tested without any real (personal or copyrighted) form:

  college_flat      underlines after labels, two fields per line, tick boxes, one-letter-per-box fields, multi-line
                    address, a "FOR OFFICE USE ONLY" block and a signature line
  medical_table     a ruled table: label cell | empty cell, and tall cells for long answers
  dotted_leaders    "Name ........................" style
  label_above_box   the label sits above an empty box
  fillable          a PDF with real fillable (AcroForm) fields
  two_page_offset   two pages and a page box that does not start at (0, 0)

``make_all(folder)`` writes them all; each ``make_*`` returns the path.
"""

from __future__ import annotations

import os
from typing import Dict

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

W, H = A4          # 595.3 x 841.9 points


def _canvas(path: str, mediabox=None) -> canvas.Canvas:
    c = canvas.Canvas(path, pagesize=A4)
    if mediabox:
        c.setPageSize((mediabox[2] - mediabox[0], mediabox[3] - mediabox[1]))
    return c


def _label_line(c, x, y, label, line_to, font=("Helvetica", 11)):
    """label at (x, y) followed by an underline to x=line_to."""
    c.setFont(*font)
    c.drawString(x, y, label)
    wd = c.stringWidth(label, *font) if isinstance(font, tuple) else 0
    c.line(x + wd + 6, y - 2, line_to, y - 2)


def make_college_flat(path: str) -> str:
    c = _canvas(path)
    c.setFont("Helvetica-Bold", 18); c.drawCentredString(W / 2, H - 60, "ABC College of Engineering")
    c.setFont("Helvetica-Bold", 13); c.drawCentredString(W / 2, H - 82, "Admission Application Form 2026")
    y = H - 130
    _label_line(c, 50, y, "Name of the Applicant *:", 545); y -= 30
    _label_line(c, 50, y, "Father's Name:", 545); y -= 30
    _label_line(c, 50, y, "Date of Birth (dd/mm/yyyy):", 330)
    _label_line(c, 345, y, "Age:", 545); y -= 30
    c.setFont("Helvetica", 11); c.drawString(50, y, "Gender:")
    x = 105
    for opt in ("Male", "Female", "Other"):
        c.rect(x, y - 1, 9, 9); c.drawString(x + 14, y, opt); x += 20 + c.stringWidth(opt, "Helvetica", 11) + 20
    _label_line(c, 345, y, "Blood Group:", 545); y -= 30
    _label_line(c, 50, y, "Mobile No. *:", 300)
    _label_line(c, 315, y, "E-mail:", 545); y -= 30
    c.setFont("Helvetica", 11); c.drawString(50, y, "Address:")
    c.line(105, y - 2, 545, y - 2); y -= 22
    c.line(50, y - 2, 545, y - 2); y -= 22
    c.line(50, y - 2, 545, y - 2); y -= 30
    c.setFont("Helvetica", 11); c.drawString(50, y, "PIN Code:")
    for i in range(6):
        c.rect(110 + i * 17, y - 4, 17, 17)
    y -= 34
    c.drawString(50, y, "Aadhaar No.:")
    for i in range(12):
        c.rect(125 + i * 17, y - 4, 17, 17)
    y -= 34
    c.drawString(50, y, "Category:")
    x = 110
    for opt in ("General", "OBC", "SC", "ST"):
        c.rect(x, y - 1, 9, 9); c.drawString(x + 14, y, opt); x += 20 + c.stringWidth(opt, "Helvetica", 11) + 18
    y -= 30
    _label_line(c, 50, y, "Course applied for:", 545); y -= 30
    _label_line(c, 50, y, "Percentage in 12th standard:", 330)
    y -= 50
    c.setFont("Helvetica-Bold", 11); c.drawString(50, y, "FOR OFFICE USE ONLY"); c.line(50, y - 3, 200, y - 3)
    y -= 28
    _label_line(c, 50, y, "Application No.:", 300)
    _label_line(c, 315, y, "Receipt No.:", 545); y -= 60
    _label_line(c, 50, y, "Signature of Applicant:", 300)
    _label_line(c, 315, y, "Date:", 545)
    c.showPage(); c.save()
    return path


def make_medical_table(path: str) -> str:
    c = _canvas(path)
    c.setFont("Helvetica-Bold", 16); c.drawCentredString(W / 2, H - 60, "City Hospital - Patient Registration")
    x0, x1, xm = 50, 545, 190
    top = H - 100
    rows = [("Patient Name", 28), ("Age", 28), ("Sex", 28), ("Mobile Number", 28), ("Blood Group", 28),
            ("Address", 70), ("Chief Complaint", 70)]
    y = top
    c.setFont("Helvetica", 11)
    for label, h in rows:
        c.rect(x0, y - h, xm - x0, h); c.rect(xm, y - h, x1 - xm, h)
        c.drawString(x0 + 6, y - 18, label)
        y -= h
    c.showPage(); c.save()
    return path


def make_dotted(path: str) -> str:
    c = _canvas(path)
    c.setFont("Helvetica-Bold", 14); c.drawCentredString(W / 2, H - 60, "Library Membership Form")
    c.setFont("Helvetica", 11)
    y = H - 120
    for label in ("Name", "Roll Number", "Department", "Phone", "Email"):
        c.drawString(50, y, label + " ")
        wd = c.stringWidth(label + " ", "Helvetica", 11)
        c.drawString(50 + wd + 4, y, "." * 70)
        y -= 32
    c.showPage(); c.save()
    return path


def make_label_above_box(path: str) -> str:
    c = _canvas(path)
    c.setFont("Helvetica-Bold", 14); c.drawCentredString(W / 2, H - 60, "Scholarship Application")
    c.setFont("Helvetica", 10)
    y = H - 110
    for label, w in (("Full Name", 300), ("Date of Birth", 140), ("Phone Number", 200), ("Annual Family Income", 200)):
        c.drawString(50, y, label)
        c.rect(50, y - 26, w, 20)
        y -= 62
    c.showPage(); c.save()
    return path


def make_fillable(path: str) -> str:
    c = _canvas(path)
    c.setFont("Helvetica-Bold", 14); c.drawCentredString(W / 2, H - 60, "Camp Registration (fillable)")
    c.setFont("Helvetica", 11)
    f = c.acroForm
    y = H - 120
    for label, name in (("Full Name:", "full_name"), ("Age:", "age"), ("Phone:", "phone"), ("Address:", "address")):
        c.drawString(50, y + 4, label)
        f.textfield(name=name, x=150, y=y - 4, width=300, height=22, borderStyle="inset", forceBorder=True,
                    tooltip=label.rstrip(":"))
        y -= 40
    c.drawString(50, y + 4, "I agree to the terms")
    f.checkbox(name="agree", x=190, y=y, size=14, tooltip="I agree to the terms", buttonStyle="check")
    c.showPage(); c.save()
    return path


def make_two_page_offset(path: str) -> str:
    c = canvas.Canvas(path, pagesize=A4)
    for page in (1, 2):
        c.setPageSize(A4)
        c.setFont("Helvetica-Bold", 14); c.drawString(50, H - 60, f"Application (page {page} of 2)")
        c.setFont("Helvetica", 11)
        if page == 1:
            _label_line(c, 50, H - 110, "Applicant Name:", 400)
            _label_line(c, 50, H - 145, "Mobile:", 300)
        else:
            _label_line(c, 50, H - 110, "Place of Birth:", 400)
            _label_line(c, 50, H - 145, "Nationality:", 300)
        c.showPage()
    c.save()
    # shift the visible box so it does not start at (0, 0): the form must still be read and filled correctly
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject
    r = PdfReader(path); w = PdfWriter(clone_from=r)
    for p in w.pages:
        p.mediabox = RectangleObject([-30, -20, W - 30, H - 20])
        p.cropbox = RectangleObject([-30, -20, W - 30, H - 20])
    with open(path, "wb") as fh:
        w.write(fh)
    return path




def make_realistic_college(path: str) -> str:
    """A fuller, official-looking admission form (border, header band, sections, notes) for demos."""
    c = _canvas(path)
    c.setLineWidth(1.2); c.rect(30, 30, W - 60, H - 60); c.setLineWidth(0.5)
    c.setFillColorRGB(0.12, 0.25, 0.5); c.rect(30, H - 100, W - 60, 70, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 17); c.drawCentredString(W / 2, H - 62, "SUNRISE INSTITUTE OF TECHNOLOGY")
    c.setFont("Helvetica", 10); c.drawCentredString(W / 2, H - 80, "Affiliated to State Technical University  |  Admissions Office")
    c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(W / 2, H - 125, "APPLICATION FOR ADMISSION - B.TECH 2026-27")
    c.setFont("Helvetica-Oblique", 8.5); c.drawString(45, H - 142, "Instructions: fill in BLOCK LETTERS. Fields marked * are compulsory. Attach self-attested copies of certificates.")
    c.rect(W - 130, H - 245, 85, 95)                                  # photo box
    c.setFont("Helvetica", 8); c.drawCentredString(W - 87, H - 200, "Affix recent")
    c.drawCentredString(W - 87, H - 210, "passport size")
    c.drawCentredString(W - 87, H - 220, "photograph")

    def section(y, title):
        c.setFillColorRGB(0.85, 0.88, 0.95); c.rect(45, y - 4, W - 90, 16, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 10); c.drawString(50, y, title)

    section(H - 170, "A. PERSONAL DETAILS")
    y = H - 200
    _label_line(c, 50, y, "1. Full Name of Candidate *:", 370, ("Helvetica", 10)); y -= 26
    _label_line(c, 50, y, "2. Father's Name:", 250, ("Helvetica", 10))
    _label_line(c, 260, y, "Mother's Name:", 450, ("Helvetica", 10)); y -= 26
    _label_line(c, 50, y, "3. Date of Birth (DD/MM/YYYY) *:", 260, ("Helvetica", 10))
    _label_line(c, 270, y, "Age:", 330, ("Helvetica", 10)); y -= 26
    c.setFont("Helvetica", 10); c.drawString(50, y, "4. Gender *:")
    x = 125
    for opt in ("Male", "Female", "Other"):
        c.rect(x, y - 1, 8, 8); c.drawString(x + 12, y, opt); x += 20 + c.stringWidth(opt, "Helvetica", 10) + 16
    _label_line(c, 300, y, "Nationality:", 450, ("Helvetica", 10)); y -= 26
    _label_line(c, 50, y, "5. Mobile Number *:", 230, ("Helvetica", 10))
    _label_line(c, 245, y, "Email ID:", 450, ("Helvetica", 10)); y -= 26
    c.drawString(50, y, "6. Permanent Address *:"); c.line(165, y - 2, 450, y - 2); y -= 20
    c.line(50, y - 2, 450, y - 2); y -= 22
    c.drawString(50, y, "PIN Code:")
    for i in range(6):
        c.rect(105 + i * 16, y - 4, 16, 16)
    c.drawString(250, y, "Aadhaar No.:")
    for i in range(12):
        c.rect(320 + i * 14, y - 4, 14, 16)
    y -= 30
    c.drawString(50, y, "7. Category:")
    x = 125
    for opt in ("General", "OBC", "SC", "ST"):
        c.rect(x, y - 1, 8, 8); c.drawString(x + 12, y, opt); x += 20 + c.stringWidth(opt, "Helvetica", 10) + 14
    y -= 38
    section(y, "B. ACADEMIC DETAILS"); y -= 28
    _label_line(c, 50, y, "8. Name of School/College last attended:", 450, ("Helvetica", 10)); y -= 26
    _label_line(c, 50, y, "9. Board / University:", 260, ("Helvetica", 10))
    _label_line(c, 270, y, "Year of passing:", 450, ("Helvetica", 10)); y -= 26
    _label_line(c, 50, y, "10. Percentage / CGPA in qualifying exam:", 330, ("Helvetica", 10)); y -= 26
    _label_line(c, 50, y, "11. Branch preferred (first choice):", 330, ("Helvetica", 10)); y -= 40
    section(y, "C. DECLARATION"); y -= 24
    c.setFont("Helvetica", 9); c.drawString(50, y, "I declare that the information given above is true to the best of my knowledge.")
    y -= 36
    _label_line(c, 50, y, "Place:", 230, ("Helvetica", 10)); _label_line(c, 260, y, "Date:", 370, ("Helvetica", 10))
    _label_line(c, 390, y, "Signature of Candidate:", 540, ("Helvetica", 9)); y -= 40
    c.setFont("Helvetica-Bold", 10); c.drawString(50, y, "FOR OFFICE USE ONLY"); c.line(50, y - 3, 170, y - 3); y -= 26
    _label_line(c, 50, y, "Application No.:", 250, ("Helvetica", 10)); _label_line(c, 270, y, "Date received:", 450, ("Helvetica", 10))
    c.showPage(); c.save()
    return path


MAKERS = {"college_flat": make_college_flat, "medical_table": make_medical_table, "dotted_leaders": make_dotted,
          "label_above_box": make_label_above_box, "fillable": make_fillable, "two_page_offset": make_two_page_offset,
          "realistic_college": make_realistic_college}


def make_all(folder: str) -> Dict[str, str]:
    os.makedirs(folder, exist_ok=True)
    return {name: fn(os.path.join(folder, name + ".pdf")) for name, fn in MAKERS.items()}


if __name__ == "__main__":
    import sys
    out = make_all(sys.argv[1] if len(sys.argv) > 1 else "sample_forms")
    print("\n".join(out.values()))


