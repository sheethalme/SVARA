"""Admin page: upload a form and choose which fields the user must fill, and what to ask for each."""
import os, sys
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import extractor
import formflow as ff

st.set_page_config(page_title="SVARA admin: form setup", layout="wide")
st.title("Admin: set up a form")

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
        st.success(f"Saved '{form['title']}' with {len(fields)} fields. Users can now fill it on the "
                   "'Fill Form by Voice' page.")
    except ValueError as e:
        st.error(str(e))

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
