import json

import pytest

import extractor
from callpage.admin_adapter import convert_admin_form, save_admin_form, sync_all_forms
from callpage.forms import FormError, FormStore


@pytest.fixture(autouse=True)
def reset_fields():
    fields = dict(extractor.FIELDS)
    aliases = dict(extractor._ALIASES)
    extractor.reset_extractor()
    yield
    extractor.FIELDS.clear()
    extractor.FIELDS.update(fields)
    extractor._ALIASES.clear()
    extractor._ALIASES.update(aliases)
    extractor.reset_extractor()


def test_adapter_preserves_order_and_canonical_names():
    form = {
        "id": "test_clinic",
        "title": "Test clinic",
        "fields": [
            {"name": "full_name", "required": True, "custom": False},
            {"name": "age_in_years", "required": True, "custom": False,
             "question": "How many years old are you?"},
            {"name": "phone_number", "required": True, "custom": False},
            {"name": "ration card number", "kind": "number", "required": False,
             "custom": True, "label": "Ration card number",
             "question": {"en": "What is your ration card number?", "hi": "आपका राशन कार्ड नंबर क्या है?"}},
        ],
    }

    data = convert_admin_form(form)

    assert [f["name"] for f in data["fields"]] == ["name", "age", "phone", "ration_card_number"]
    assert data["fields"][1]["question"] == "How many years old are you?"
    assert data["fields"][3]["kind"] == "number"
    assert data["fields"][3]["required"] is False
    assert data["fields"][3]["question"]["hi"] == "आपका राशन कार्ड नंबर क्या है?"


def test_save_admin_form_writes_valid_json(tmp_path):
    form = {
        "id": "test_clinic",
        "title": "Test clinic",
        "language": "en",
        "languages": ["en", "hi"],
        "fields": [
            {"name": "name", "required": True},
            {"name": "age", "required": True, "question": "How many years old are you?"},
            {"name": "phone", "required": True},
            {"name": "ration card number", "kind": "text", "required": True,
             "custom": True, "label": "Ration card number",
             "question": "What is your ration card number?"},
        ],
    }

    spec = save_admin_form(form, str(tmp_path))

    path = tmp_path / "test_clinic.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["fields"][0]["name"] == "name"
    assert [f.name for f in spec.fields] == ["name", "age", "phone", "ration_card_number"]


def test_adapter_rejects_invalid_input():
    with pytest.raises(FormError, match="title"):
        convert_admin_form({"fields": [{"name": "name"}]})
    with pytest.raises(FormError, match="at least one"):
        convert_admin_form({"title": "Empty", "fields": []})
    with pytest.raises(FormError, match="'name' is missing"):
        convert_admin_form({"title": "Bad", "fields": [{}]})


def test_sync_all_forms_exports_existing_admin_folders(tmp_path):
    source = tmp_path / "existing"
    old = source / "12charsid123"
    old.mkdir(parents=True)
    (old / "form.json").write_text(json.dumps({
        "id": "12charsid123",
        "title": "Old clinic",
        "fields": [
            {"name": "name", "label": "Full name", "question": "Please say your name.", "required": True},
            {"name": "ward_number", "label": "Ward number", "question": "What ward are you in?",
             "required": False, "kind": "number", "custom": True},
        ],
    }), encoding="utf-8")

    output = tmp_path / "call_forms"
    saved = sync_all_forms(str(output), source_dir=str(source))

    assert len(saved) == 1
    assert saved[0].id == "12charsid123"
    assert [f.name for f in saved[0].fields] == ["name", "ward_number"]
    assert FormStore(str(output)).get("12charsid123").fields[1].required is False
