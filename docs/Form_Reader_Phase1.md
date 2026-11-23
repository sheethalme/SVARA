# SVARA Form Reader — Phase 1 (digital PDF forms, English, fully local)

Upload a blank digital PDF form (college, hospital, ...). SVARA finds the places to fill, understands each label,
turns them into the questions of a voice call, stores the answers, and writes them onto the **original PDF**.
**Nothing is sent to the cloud for reading or filling the form** (no network library is imported in `formreader/`; a test checks this).
Speech recognition for the call itself is a separate setting.

## Flow
1. **Analyse** `analyze_pdf(pdf)` -> `FormTemplate` (fields, boxes, meanings, confidence, warnings)
2. **Review** (admin screen): preview with numbered boxes; fix label / call name / required / ignore; add missing fields
3. **Publish** `publish_template(template, FormStore)` -> call form (form id = template id); the call asks the detected questions
4. **Call** -> stored in the existing call records
5. **Fill** `fill_from_record(template, pdf_bytes, record)` -> filled PDF + report; download

## What it detects
| Layout | How |
|---|---|
| Fillable PDF (AcroForm) | real widgets; tooltip or nearby text is the label |
| `Label: ______` and `Label ........` | underline / dot-leader blank, label on its left |
| Label above an empty box | closest text above |
| Ruled tables (label cell, empty cell) | table cells; tall cells -> multi-line |
| One-letter-per-box (PIN, Aadhaar, phone) | equal touching squares -> comb field |
| Tick boxes (`[ ] Male [ ] Female`) | one choice field with options |
| Multi-line (address) | extra blank lines under one label |
| "FOR OFFICE USE ONLY" | ignored until the applicant's signature line |
| Signature / photo | ignored |

Meaning is by plain regex rules (`semantics.py`): name, age, gender, phone, date of birth, address, PIN, Aadhaar, e-mail, blood group,
number, yes/no, else text. Father's/Mother's/Permanent... get their own names. Questions are generated in English.

## Honest limits (tested only on generated sample forms so far)
* Real forms vary a lot: **the review step is essential**. Low-confidence fields are flagged orange.
* Scans/photos are not read (a warning is shown); OCR is Phase 2. Rotated pages, AcroForm radio groups, repeated table rows: partial.
* Only English text is written (standard PDF font); other characters are dropped with a warning.
* Choice answers are matched to options by fuzzy text at fill time; unmatched ones are left blank and reported.
* Filling flat PDFs is verified by an independent renderer in the tests; always look at the preview before using the result.

## Try it on your own form
```
python -m formreader analyze my_form.pdf --out form_analysis
```
Prints the fields and writes `template.json` + `page1_fields.png` (boxes drawn on the form).
Fill test: `python -m formreader fill my_form.pdf answers.json out.pdf` (answers = `{"name": "...", ...}` by call name).

Privacy: templates and blank PDFs are stored in `data/templates/`, filled PDFs contain personal data: keep them out of git
(`merge_formreader.py` adds the ignore lines).
