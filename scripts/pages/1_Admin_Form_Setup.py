"""Admin page: upload a form and choose which fields the user must fill, and what to ask for each."""
import os, sys
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import extractor
import formflow as ff
import tts
from callpage.admin_adapter import save_admin_form, sync_all_forms
from callpage.forms import FormError, FormStore
from formreader.analyze import analyze_pdf
from formreader.model import TemplateError
from formreader.publish import publish_template
from formreader.render import render_overlay
from formreader.store import TemplateStore

st.set_page_config(page_title="SVARA admin: form setup", layout="wide")
st.title("Admin: set up a form")

try:
    sync_all_forms()
except FormError as e:
    st.error(str(e))

PIN = os.environ.get("SVARA_ADMIN_PIN")                       # optional: set it in .env to lock this page
if PIN and not st.session_state.get("admin_ok"):
    if st.text_input("Admin PIN", type="password") == PIN:
        st.session_state["admin_ok"] = True; st.rerun()
    st.stop()

custom = st.session_state.setdefault("custom_fields", {})   # name -> field dict (this session)
builtin = [n for n in extractor.list_fields() if n not in custom]

st.subheader("1. The form")
c1, c2 = st.columns([1, 2])
title = c1.text_input("Form title", placeholder="e.g. Clinic registration")
up = c2.file_uploader("Upload the form (optional: PDF, image or Word)", type=[e.strip(".") for e in sorted(ff.ALLOWED_UPLOADS)])
if up is not None:
    if up.type and up.type.startswith("image/"):
        st.image(up, caption=up.name, width=360)
    else:
        st.caption(f"Attached: {up.name} ({up.size // 1024} KB). It is stored with the form for reference.")

st.subheader("2. Fields the user must fill")
chosen = st.multiselect("Pick the fields, in the order they should be asked", builtin + list(custom),
                        format_func=lambda n: n.replace("_", " "))
with st.expander("The field you need is not in the list? Add your own"):
    a, b, c = st.columns(3)
    new_label = a.text_input("Field label", placeholder="e.g. Ward number")
    new_kind = b.selectbox("What kind of answer is it?", sorted(extractor.RULES),
                           index=sorted(extractor.RULES).index("text"),
                           help="text = free words; number = digits; name; date; yes_no; ...")
    new_q = c.text_input("Question to ask", placeholder="Which ward do you live in?")
    if st.button("Add this field"):
        if not new_label.strip():
            st.error("Give the field a label.")
        else:
            f = ff.make_field(new_label, new_q, custom=True, kind=new_kind, label=new_label.strip())
            if f["name"] in extractor.list_fields() and f["name"] not in custom:
                st.error("That name is already a built-in field; pick it from the list above.")
            else:
                ff.register_form_fields({"fields": [f]}); custom[f["name"]] = f
                st.success(f"Added '{f['label']}'. Now select it in the list above."); st.rerun()

st.subheader("3. Questions")
rows = []
for n in chosen:
    f = custom.get(n) or ff.make_field(n)
    rows.append({"Field": f["label"], "Question asked aloud": f["question"], "Required": True, "_name": n})
if rows:
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, width="stretch", key="ed_" + "|".join(chosen),
        disabled=["Field"], column_config={"_name": None, "Required": st.column_config.CheckboxColumn()})
else:
    st.info("Choose at least one field above.")
    edited = None

if st.button("Save form", type="primary", disabled=edited is None):
    fields = []
    for _, r in edited.iterrows():
        base = custom.get(r["_name"])
        fields.append(ff.make_field(r["_name"], r["Question asked aloud"], bool(r["Required"]),
                                    custom=bool(base), kind=(base or {}).get("kind", "text"),
                                    label=(base or {}).get("label", "")))
    try:
        form = ff.save_form(title, fields, up.getvalue() if up else None, up.name if up else None)
        save_admin_form(form)
        st.success(f"Saved '{form['title']}' with {len(fields)} fields. Users can now fill it on the "
                   "'Fill Form by Voice' page.")
        with st.spinner("Preparing the spoken questions (2C text-to-speech)..."):
            pre = tts.prefetch([f["question"] for f in fields], "en")
        if pre["errors"]:
            st.info("Spoken questions could not be prepared now (no internet?); they will be made the first "
                    "time a user hears them, or read by the browser's voice.")
    except (ValueError, FormError) as e:
        st.error(str(e))

st.divider()
st.subheader("Upload a real PDF form")

PDF_TEMPLATE_DIR = os.path.join(ROOT, "data", "templates")
template_store = TemplateStore(PDF_TEMPLATE_DIR)
valid_pdf_kinds = [
    "name", "age", "gender", "phone", "date_of_birth", "date", "address",
    "pincode", "aadhaar", "email", "blood_group", "number", "yes_no", "text",
]

pdf_upload = st.file_uploader("Choose a PDF form", type=["pdf"], key="pdf_form_upload")
if pdf_upload is not None:
    pdf_bytes = pdf_upload.getvalue()
    pdf_key = f"{pdf_upload.name}:{len(pdf_bytes)}"
    if st.session_state.get("pdf_template_source_key") != pdf_key:
        with st.spinner("Reading the PDF locally..."):
            template = analyze_pdf(pdf_bytes, filename=pdf_upload.name)
        st.session_state["pdf_template"] = template
        st.session_state["pdf_template_bytes"] = pdf_bytes
        st.session_state["pdf_template_source_key"] = pdf_key
        st.session_state["pdf_template_title"] = template.title

template = st.session_state.get("pdf_template")
pdf_template_bytes = st.session_state.get("pdf_template_bytes")

if template is not None and pdf_template_bytes is not None:
    for warning in template.warnings:
        st.warning(warning)

    template_title = st.text_input(
        "PDF form title",
        key="pdf_template_title",
    )
    template.title = template_title

    page_count = len(template.pages)
    selected_page = st.selectbox(
        "Preview page",
        range(page_count),
        format_func=lambda p: f"Page {p + 1}",
        key="pdf_template_preview_page",
    )
    st.image(render_overlay(pdf_template_bytes, template, selected_page), caption="Numbered PDF field preview")
    st.caption("Legend: green = sure | orange = please check | grey = ignored")

    review_rows = []
    for index, field in enumerate(template.fields, 1):
        review_rows.append({
            "#": index,
            "Label": field.label,
            "Call name": field.call_name,
            "Kind": field.call_kind,
            "Required": field.required,
            "Ignore": field.ignored,
            "Question": field.question or "",
            "Shape": field.shape,
            "Confidence": round(field.confidence, 2),
            "Notes": "; ".join(field.notes),
        })

    reviewed = st.data_editor(
        pd.DataFrame(review_rows),
        hide_index=True,
        width="stretch",
        key=f"pdf_template_editor_{template.id}",
        disabled=["#", "Shape", "Confidence", "Notes"],
        column_config={
            "Kind": st.column_config.SelectboxColumn(options=valid_pdf_kinds),
            "Required": st.column_config.CheckboxColumn(),
            "Ignore": st.column_config.CheckboxColumn(),
        },
    )

    for field, row in zip(template.fields, reviewed.to_dict("records")):
        field.label = str(row["Label"] or "").strip()
        field.call_name = str(row["Call name"] or "").strip()
        field.call_kind = str(row["Kind"] or "text")
        field.required = bool(row["Required"])
        field.ignored = bool(row["Ignore"])
        field.question = str(row["Question"] or "").strip() or None

    validation_error = None
    try:
        template.validate()
        active_names = [f.call_name for f in template.active_fields()]
        if len(active_names) != len(set(active_names)):
            validation_error = "Call names must be unique. Two active fields use the same call name."
        bad_kind = [f.label for f in template.active_fields() if f.call_kind not in valid_pdf_kinds]
        if bad_kind:
            validation_error = f"Invalid call kind on: {', '.join(bad_kind)}"
    except TemplateError as exc:
        validation_error = str(exc)
    if validation_error:
        st.error(validation_error)

    save_template, publish_template_button = st.columns(2)
    if save_template.button("Save template", disabled=bool(validation_error)):
        try:
            template_store.save(template, pdf_template_bytes)
            st.success("Template saved locally.")
        except (TemplateError, OSError) as exc:
            st.error(str(exc))

    if publish_template_button.button("Publish for voice call", disabled=bool(validation_error)):
        try:
            template_store.save(template, pdf_template_bytes)
            spec = publish_template(template, FormStore(os.path.join(ROOT, "data", "forms")))
            # Keep the template in sync if publishing canonicalises a field name.
            template_store.save(template, pdf_template_bytes)
            st.success(
                f"Published '{spec.title}'. It is now available in 'Fill Form by Voice'; "
                "the call will ask the detected questions."
            )
        except (TemplateError, FormError, OSError) as exc:
            st.error(str(exc))

st.divider()
st.subheader("Saved PDF templates")
saved_templates = template_store.list()
if not saved_templates:
    st.caption("No PDF templates saved yet.")
for template_id, template_title in saved_templates:
    with st.expander(f"{template_title} ({template_id})"):
        load_template = st.button("Load", key=f"load_pdf_template_{template_id}")
        if load_template:
            try:
                loaded = template_store.load(template_id)
                st.session_state["pdf_template"] = loaded
                st.session_state["pdf_template_bytes"] = template_store.pdf_bytes(template_id)
                st.session_state["pdf_template_source_key"] = f"saved:{template_id}"
                st.session_state["pdf_template_title"] = loaded.title
                st.rerun()
            except (TemplateError, OSError) as exc:
                st.error(str(exc))
        confirm_delete = st.checkbox("Confirm deletion", key=f"confirm_delete_pdf_template_{template_id}")
        if st.button("Delete", disabled=not confirm_delete, key=f"delete_pdf_template_{template_id}"):
            if template_store.delete(template_id):
                st.success("PDF template deleted.")
                st.rerun()

st.divider()
st.subheader("Saved forms")
forms = ff.list_forms()
if not forms:
    st.caption("No forms yet.")
for f in forms:
    with st.expander(f"{f['title']}  ({f['created']}, {len(f['fields'])} fields)"):
        st.table([{"#": i + 1, "Field": x["label"], "Question": x["question"],
                   "Required": "yes" if x["required"] else "no"} for i, x in enumerate(f["fields"])])
        p = ff.original_path(f)
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                st.download_button(f"Download uploaded file ({f['original_name']})", fh.read(),
                                   file_name=f["original_name"], key="dl_" + f["id"])
        if st.button("Delete this form", key="del_" + f["id"]):
            ff.delete_form(f["id"]); st.rerun()

