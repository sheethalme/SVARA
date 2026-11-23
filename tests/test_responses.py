import csv
import io
import json
from types import SimpleNamespace

import pytest

from callpage.forms import FormStore, build_form_dict
from callpage.responses import (
    add_correction,
    delete_call,
    delete_form_calls,
    export_csv,
    filter_records,
    group_calls,
    initial_form_id,
    paginate,
    response_rows,
    summarize,
)
from callpage.store import CallStore


def make_record(call_id, form_id="clinic", title="Clinic", outcome="completed",
                language="hi", started="2026-01-10T10:00:00+00:00",
                name="अनिल", phone="9876543210", status="confirmed"):
    return {
        "call_id": call_id,
        "form_id": form_id,
        "form_title": title,
        "language": language,
        "started_at": started,
        "ended_at": started,
        "outcome": outcome,
        "consent": True,
        "progress": [2, 2],
        "fields": [
            {"name": "name", "label": "नाम", "kind": "name", "required": True,
             "sensitive": False, "status": status, "value": name, "display": name,
             "confidence": 0.9, "method": "rules", "attempts": 1, "rejections": 0,
             "raw_answers": [name], "confirmed_at": started},
            {"name": "phone", "label": "Phone", "kind": "phone", "required": True,
             "sensitive": True, "status": status, "value": phone, "display": phone,
             "confidence": 0.8, "method": "rules", "attempts": 1, "rejections": 0,
             "raw_answers": [phone], "confirmed_at": started},
        ],
        "events": [{"t": started, "role": "user", "text": name, "phase": "ask", "field": "name"}],
        "settings": {},
    }


@pytest.fixture
def folders(tmp_path):
    calls = tmp_path / "calls"
    forms = tmp_path / "forms"
    calls.mkdir()
    forms.mkdir()
    return calls, forms


def test_grouping_removed_forms_summary_and_column_order(folders):
    calls_dir, forms_dir = folders
    store = CallStore(str(calls_dir))
    forms = FormStore(str(forms_dir))
    forms.save(build_form_dict("Clinic", ["name", "phone"], form_id="clinic"))

    store.save(make_record("aaaaaaaaaaaaaaaa", outcome="completed"))
    store.save(make_record("bbbbbbbbbbbbbbbb", form_id="gone", title="Old Form",
                           outcome="timed_out"))
    groups = group_calls(store, forms)
    assert set(groups) == {"clinic", "gone"}
    assert "Old Form (form removed)" in groups["gone"]["title"]

    summary = summarize([make_record("a", outcome="completed"),
                         make_record("b", outcome="needs_help"),
                         make_record("c", outcome="hangup")])
    assert summary["total"] == 3
    assert summary["completed"] == 1
    assert summary["needs_help"] == 1
    assert summary["timed_out_hangup_declined"] == 1
    assert summary["partial"] == 2

    spec = forms.get("clinic")
    rows = response_rows([make_record("cccccccccccccccc")], spec)
    assert list(rows[0])[-2:] == ["name", "phone"]


def test_filters_skip_empty_and_mask(folders):
    calls_dir, forms_dir = folders
    store = CallStore(str(calls_dir))
    forms = FormStore(str(forms_dir))
    spec = forms.save(build_form_dict("Clinic", ["name", "phone"], form_id="clinic"))
    records = [
        make_record("aaaaaaaaaaaaaaaa", outcome="completed", language="hi"),
        make_record("bbbbbbbbbbbbbbbb", outcome="needs_help", language="kn",
                    name="No Name", phone="", status="skipped"),
    ]

    assert len(filter_records(records, outcome="Completed")) == 1
    assert len(filter_records(records, outcome="All", language="kn")) == 1
    assert len(filter_records(records, outcome="All", start_date="2026-01-10",
                              end_date="2026-01-10")) == 2
    assert len(filter_records(records, outcome="All", search="987654")) == 1

    rows = response_rows(records, spec, mask_sensitive=True)
    assert rows[0]["phone"] == "********10"
    assert rows[1]["name"] == "(skipped)"
    assert rows[1]["phone"] == "(skipped)"
    revealed = response_rows(records, spec, mask_sensitive=False)
    assert revealed[0]["phone"] == "9876543210"
    assert len(filter_records(records, outcome="All")) == 2


def test_csv_bom_hindi_and_unicode_round_trip(folders):
    calls_dir, forms_dir = folders
    store = CallStore(str(calls_dir))
    forms = FormStore(str(forms_dir))
    spec = forms.save(build_form_dict("Clinic", ["name", "phone"], form_id="clinic"))
    record = make_record("aaaaaaaaaaaaaaaa", name="अनिल कुमार")

    csv_text = export_csv([record], spec)
    assert csv_text.startswith("\ufeff")
    parsed = list(csv.DictReader(io.StringIO(csv_text)))
    assert parsed[0]["name"] == "अनिल कुमार"


def test_correction_validation_original_audit_and_refresh(folders):
    calls_dir, forms_dir = folders
    store = CallStore(str(calls_dir))
    record = make_record("aaaaaaaaaaaaaaaa")
    store.save(record)

    fake = SimpleNamespace(
        extract_field_full=lambda text, field: SimpleNamespace(
            status="ok", value="सुनील", display="सुनील", issues=[]
        )
    )
    updated = add_correction(store, "aaaaaaaaaaaaaaaa", "name", "सुनील", ex=fake)
    assert updated["reviewed"] is True
    assert updated["corrections"][0]["old"] == "अनिल"
    assert updated["corrections"][0]["new"] == "सुनील"
    assert updated["corrections"][0]["by"] == "admin"
    assert updated["fields"][0]["value"] == "अनिल"

    loaded = store.load("aaaaaaaaaaaaaaaa")
    assert response_rows([loaded])[0]["name"] == "सुनील ✎"

    bad = SimpleNamespace(
        extract_field_full=lambda text, field: SimpleNamespace(
            status="invalid", value=None, issues=["Phone is invalid"]
        )
    )
    with pytest.raises(ValueError, match="Phone is invalid"):
        add_correction(store, "aaaaaaaaaaaaaaaa", "phone", "bad", ex=bad)

    in_progress = make_record("bbbbbbbbbbbbbbbb", outcome="in_progress")
    store.save(in_progress)
    with pytest.raises(ValueError, match="in-progress"):
        add_correction(store, "bbbbbbbbbbbbbbbb", "name", "changed", ex=fake)


def test_unreadable_files_do_not_break_grouping(folders):
    calls_dir, forms_dir = folders
    (calls_dir / "abcdef0123456789.json").write_text("{broken")
    store = CallStore(str(calls_dir))
    assert group_calls(store, FormStore(str(forms_dir))) == {}


def test_initial_form_is_most_recent_call(folders):
    calls_dir, forms_dir = folders
    store = CallStore(str(calls_dir))
    old = make_record("aaaaaaaaaaaaaaaa", form_id="old", started="2026-01-10T10:00:00+00:00")
    new = make_record("bbbbbbbbbbbbbbbb", form_id="new", started="2026-11-23T10:00:00+00:00")
    store.save(old)
    store.save(new)
    groups = group_calls(store, FormStore(str(forms_dir)))
    assert initial_form_id(groups) == "new"


def test_delete_one_and_all_form_calls(folders):
    calls_dir, forms_dir = folders
    store = CallStore(str(calls_dir))
    store.save(make_record("aaaaaaaaaaaaaaaa", form_id="clinic"))
    store.save(make_record("bbbbbbbbbbbbbbbb", form_id="clinic"))
    store.save(make_record("cccccccccccccccc", form_id="other"))

    assert delete_call(store, "aaaaaaaaaaaaaaaa") is True
    assert store.load("bbbbbbbbbbbbbbbb")["call_id"] == "bbbbbbbbbbbbbbbb"
    assert delete_form_calls(store, "clinic") == 1
    assert len(store.list()) == 1


def test_pagination_boundaries():
    records = [{"call_id": str(i)} for i in range(51)]
    first = paginate(records, 1, 25)
    second = paginate(records, 2, 25)
    last = paginate(records, 99, 25)
    assert len(first["rows"]) == 25 and first["pages"] == 3
    assert len(second["rows"]) == 25 and second["page"] == 2
    assert len(last["rows"]) == 1 and last["page"] == 3
