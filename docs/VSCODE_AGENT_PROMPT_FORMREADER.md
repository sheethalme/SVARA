# Prompt for the VS Code agent (copy everything below the line)

---
Integrate the new `formreader/` package (already in the project root, with tests in `tests/test_formreader.py`) into the Streamlit app. Do NOT commit or push. Do NOT send any PDF to a cloud service; everything must stay local. Do not modify `formreader/` except to fix a real bug (tell me if you do).

First run: `pip install -r requirements.txt` then `python -m pytest tests/test_formreader.py -q` (must pass).

API (read the docstrings): `formreader.analyze.analyze_pdf(bytes, filename=...) -> FormTemplate`; `formreader.render.render_overlay(pdf_bytes, template, page) -> PNG bytes`; `formreader.store.TemplateStore("data/templates")` (`save(template, pdf_bytes)`, `load`, `pdf_bytes`, `list`, `delete`); `formreader.publish.publish_template(template, FormStore("data/forms"))` (raises `callpage.forms.FormError`); `formreader.publish.fill_from_record(template, pdf_bytes, call_record) -> (pdf_bytes, report)`; `formreader.semantics.infer_kind/make_question`; `TField` fields: id, label, page, bbox, shape, call_kind, call_name, required, ignored, question, options, confidence, notes.

1. In the existing "Admin Form Setup" page add a section "Upload a real PDF form": `st.file_uploader(type="pdf")` -> analyse (spinner) -> show `template.warnings`, and the page preview from `render_overlay` (page selector if several pages; legend: green sure, orange check, grey ignored).
2. Review table (`st.data_editor`, one row per field, numbered as in the preview): editable Label, Call name (lower-case letters/digits/_, unique), Kind (extractor kinds: name, age, gender, phone, date_of_birth, date, address, pincode, aadhaar, email, blood_group, number, yes_no, text), Required, Ignore, Question (blank = auto via `make_question`). Show a read-only Shape and Confidence column. Apply the edits back to the `TField`s before publishing; recompute preview.
3. Form title text input (default `template.title`) and form id; buttons: "Save template" (TemplateStore.save with the PDF bytes) and "Publish for voice call" (publish_template; show FormError text with st.error; on success say the form is available in "Fill Form by Voice" and the call will ask the detected questions). Keep `st.session_state` for the template so reruns do not re-analyse.
4. List saved templates with Load and Delete (confirm before delete).
5. On the Form Responses page (results stored per form), for forms that have a saved template (TemplateStore id == form id): per response a "Preview filled PDF" (render_page of the filled bytes) and `st.download_button("Download filled PDF")` (file name `<form_id>_<call_id>.pdf`), and show `report["warnings"]` and skipped fields so the user can see what was left blank. Also a "Download all as ZIP" is optional. Filled PDFs are generated on demand and not stored.
6. If a record has corrections, `answers_from_record` already uses the latest value; do not change its format.
7. Add a short README note and keep all new code in the existing Streamlit pages/utility modules (no new network calls).

When done, run the whole test suite and tell me what you changed. Then upload one of my real PDF forms to test the flow and tell me what was detected wrongly.
