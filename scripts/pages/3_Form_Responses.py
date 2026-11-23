"""Streamlit page for browsing and correcting saved form responses."""
import os
import sys

import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)

from callpage.forms import FormStore
import callpage.responses as response_logic
from callpage.responses import (
    add_correction,
    audio_turns,
    delete_call,
    delete_form_calls,
    export_csv,
    export_json,
    filter_records,
    group_calls,
    paginate,
    summarize,
    to_rows,
)
from callpage.store import CallStore
from formreader.publish import fill_from_record
from formreader.render import render_page
from formreader.store import TemplateStore


st.set_page_config(page_title="SVARA: form responses", layout="wide")
st.title("Form Responses")

PIN = os.environ.get("SVARA_ADMIN_PIN")
if PIN and not st.session_state.get("admin_ok"):
    if st.text_input("Admin PIN", type="password") == PIN:
        st.session_state["admin_ok"] = True
        st.rerun()
    st.stop()

FORMS_DIR = os.path.join(ROOT, "data", "forms")
CALLS_DIR = os.path.join(ROOT, "data", "calls")
form_store = FormStore(FORMS_DIR)
call_store = CallStore(CALLS_DIR)

top = st.columns([5, 1])
with top[1]:
    if st.button("Refresh"):
        st.rerun()

groups = group_calls(call_store, form_store)
if not groups:
    st.info("No forms or saved calls.")
    st.stop()

form_ids = list(groups)
preferred_id = response_logic.initial_form_id(groups)
if preferred_id not in form_ids:
    preferred_id = form_ids[0]
if "responses_selected_form" not in st.session_state:
    st.session_state["responses_selected_form"] = preferred_id
elif st.session_state["responses_selected_form"] not in form_ids:
    st.session_state["responses_selected_form"] = preferred_id
selected_id = st.selectbox(
    "Choose a form",
    form_ids,
    format_func=lambda fid: groups[fid]["title"],
    index=form_ids.index(st.session_state["responses_selected_form"]),
    key="responses_selected_form",
)
group = groups[selected_id]
spec = group["spec"]

summaries = list(group["calls"])
summary = summarize(summaries)
metric = st.columns(5)
metric[0].metric("Total calls", summary["total"])
metric[1].metric("Completed", summary["completed"])
metric[2].metric("Needs help", summary["needs_help"])
metric[3].metric("Timed out / hung up / declined", summary["timed_out_hangup_declined"])
metric[4].metric("Completion rate", f"{summary['completion_rate']:.1f}%")

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
outcome = c1.selectbox("Outcome", ["Completed", "All"], index=0)
languages = sorted({r.get("language") for r in summaries if r.get("language")})
language = c2.selectbox("Language", ["All"] + languages)
start_date = c3.date_input("From date", value=None)
end_date = c4.date_input("To date", value=None)
search = c5.text_input("Search values")

show_sensitive = st.checkbox("Show sensitive values", value=False)

filtered_summaries = filter_records(
    summaries,
    outcome=outcome,
    start_date=start_date,
    end_date=end_date,
    language=language,
    search="",
)
if search.strip():
    filtered_records = []
    for summary in filtered_summaries:
        try:
            filtered_records.append(call_store.load(summary["call_id"]))
        except Exception:
            continue
    filtered = filter_records(
        filtered_records,
        outcome=outcome,
        start_date=start_date,
        end_date=end_date,
        language=language,
        search=search,
    )
else:
    filtered = filtered_summaries
filtered_full = []
for filtered_record in filtered:
    try:
        filtered_full.append(call_store.load(filtered_record["call_id"]))
    except Exception:
        continue
page_size = 25
page_key = f"responses_page_{selected_id}"
page = st.session_state.setdefault(page_key, 1)
page_data = paginate(filtered, page=page, page_size=page_size)
if page_data["page"] != page:
    st.session_state[page_key] = page_data["page"]

page_records = []
for summary in page_data["rows"]:
    try:
        page_records.append(call_store.load(summary["call_id"]))
    except Exception:
        continue
rows = to_rows(page_records, spec, mask_sensitive=not show_sensitive)
if spec is not None:
    labels = {}
    for name, label, sensitive in __import__("callpage.responses", fromlist=["field_columns"]).field_columns(spec):
        labels[name] = f"{label}{' *' if sensitive else ''}"
    display_rows = []
    for row in rows:
        display_rows.append({
            "time": row["time"],
            "language": row["language"],
            "outcome": row["outcome"],
            "answered": row["answered"],
            **{labels[name]: row.get(name, "") for name in [f.name for f in spec.fields]},
        })
else:
    display_rows = [
        {k: v for k, v in row.items() if k not in ("call_id",)}
        for row in rows
    ]

if display_rows:
    st.dataframe(display_rows, hide_index=True, width="stretch")
else:
    st.caption("No responses match these filters.")

nav1, nav2, nav3 = st.columns([1, 2, 1])
with nav1:
    if st.button("Previous", disabled=page_data["page"] <= 1):
        st.session_state[page_key] -= 1
        st.rerun()
with nav2:
    st.caption(f"Page {page_data['page']} of {page_data['pages']} ({page_data['total']} responses)")
with nav3:
    if st.button("Next", disabled=page_data["page"] >= page_data["pages"]):
        st.session_state[page_key] += 1
        st.rerun()

st.download_button(
    "Download CSV",
    export_csv(filtered_full, spec, mask_sensitive=not show_sensitive),
    file_name=f"{selected_id}_responses.csv",
    mime="text/csv",
)

st.divider()
call_ids = [r["call_id"] for r in page_records]
if call_ids:
    detail_id = st.selectbox(
        "Select a call to view",
        call_ids,
        format_func=lambda cid: next(
            (f"{r.get('started_at') or ''} - {cid}" for r in page_records if r["call_id"] == cid),
            cid,
        ),
    )
else:
    detail_id = None

if detail_id:
    selected = call_store.load(detail_id)
    st.subheader("Call details")
    field_by_name = {f["name"]: f for f in selected.get("fields", [])}
    names = [f.name for f in spec.fields] if spec is not None else list(field_by_name)
    for name in names:
        f = field_by_name.get(name)
        if f is None:
            continue
        corrections = [c for c in selected.get("corrections", []) if c.get("field") == name]
        final = corrections[-1]["new"] if corrections else f.get("value")
        if show_sensitive is False and f.get("sensitive") and final not in (None, ""):
            shown = "*" * max(0, len(str(final)) - 2) + str(final)[-2:]
        else:
            shown = final
        label = spec.field_by_name(name).label_for(selected.get("language") or "en") if spec else f.get("label", name)
        st.write(f"**{label}**: {shown if shown not in (None, '') else '(empty)'}")
        st.caption(
            f"Status: {f.get('status') or 'pending'} | Confidence: {f.get('confidence', 0)} | "
            f"Method: {f.get('method') or '-'} | Attempts: {f.get('attempts', 0)} | "
            f"Confirmed: {f.get('confirmed_at') or '-'}"
        )
        if corrections:
            st.caption(f"Original: {f.get('value') if f.get('value') is not None else '(empty)'}")
        if selected.get("outcome") != "in_progress":
            with st.form(key=f"correction_{detail_id}_{name}", clear_on_submit=True):
                new_text = st.text_input("Corrected value", key=f"correction_text_{detail_id}_{name}")
                submitted = st.form_submit_button("Validate and save correction")
                if submitted:
                    if not new_text.strip():
                        st.error("Enter a corrected value.")
                    else:
                        try:
                            add_correction(call_store, detail_id, name, new_text)
                            st.success("Correction saved.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

    with st.expander("Conversation"):
        st.dataframe(selected.get("events", []), hide_index=True, width="stretch")

    for path in audio_turns(call_store, detail_id):
        st.caption(os.path.basename(path))
        with open(path, "rb") as fh:
            st.audio(fh.read(), format="audio/wav")

    st.download_button(
        "Download this response as JSON",
        export_json(selected),
        file_name=f"{detail_id}.json",
        mime="application/json",
    )

    template_store = TemplateStore(os.path.join(ROOT, "data", "templates"))
    try:
        template = template_store.load(selected_id)
        blank_pdf = template_store.pdf_bytes(selected_id)
        filled_pdf, fill_report = fill_from_record(template, blank_pdf, selected)
        st.divider()
        st.subheader("Filled PDF")
        for warning in fill_report.get("warnings", []):
            st.warning(warning)
        for skipped in fill_report.get("skipped", []):
            st.caption(f"Skipped: {skipped.get('label') or skipped.get('field')} - {skipped.get('why')}")
        st.download_button(
            "Download filled PDF",
            filled_pdf,
            file_name=f"{selected_id}_{detail_id}.pdf",
            mime="application/pdf",
        )
        for page_number in range(len(template.pages)):
            st.image(
                render_page(filled_pdf, page_number),
                caption=f"Filled PDF preview - page {page_number + 1}",
            )
    except Exception as exc:
        if selected_id in {template_id for template_id, _ in template_store.list()}:
            st.error(f"Could not generate the filled PDF: {exc}")

    st.divider()
    confirm_delete = st.checkbox("Confirm deletion", key=f"confirm_delete_{detail_id}")
    if st.button("Delete selected call", disabled=not confirm_delete):
        delete_call(call_store, detail_id)
        st.success("Call deleted.")
        st.rerun()

st.divider()
st.subheader("Delete all responses")
title = group["title"].replace(" (form removed)", "")
typed = st.text_input("Type the form title to confirm", key=f"delete_all_title_{selected_id}")
if st.button("Delete all responses of this form", disabled=typed != title):
    delete_form_calls(call_store, selected_id)
    st.success("All responses for this form were deleted.")
    st.rerun()
