"""Tests for callpage prompts (all four languages), forms (the admin's JSON) and the call store."""
import json
import os
import re
import string

import pytest

import extractor
from callpage import prompts
from callpage.forms import FormError, FormSpec, FormStore
from callpage.store import CallStore, StoreError

HERE = os.path.dirname(__file__)
RANGES = {"hi": (0x0900, 0x097F), "ml": (0x0D00, 0x0D7F), "kn": (0x0C80, 0x0CFF)}


def scripts_in(text):
    found = set()
    for c in text:
        cp = ord(c)
        for name, (lo, hi) in RANGES.items():
            if lo <= cp <= hi:
                found.add(name)
    return found


# ====================================================================== prompts
def test_every_sentence_exists_in_all_four_languages():
    for key, table in prompts.SENTENCES.items():
        assert set(table) == set(prompts.LANGS), key
        assert all(v.strip() for v in table.values()), key


def test_placeholders_match_across_languages():
    for key, table in prompts.SENTENCES.items():
        sets = {lang: set(re.findall(r"\{(\w+)\}", text)) for lang, text in table.items()}
        assert len({frozenset(s) for s in sets.values()}) == 1, (key, sets)


@pytest.mark.parametrize("lang", ["hi", "ml", "kn"])
def test_no_mixed_indic_scripts_in_any_sentence(lang):
    """The earlier tests found a Devanagari letter typed inside a Malayalam word; this guards against that class of typo."""
    texts = [t[lang] for t in prompts.SENTENCES.values()] + [t[lang] for t in prompts.QUESTIONS.values()] \
        + [t[lang] for t in prompts.LABELS.values()] + [prompts.CONFIRM_HINT[lang]]
    for t in texts:
        s = scripts_in(t)
        assert s <= {lang}, (t, s)
        has_words = any(c.isalpha() for c in re.sub(r"\{\w+\}", "", t))
        assert lang in s or not has_words, t                      # every Indic sentence really contains its own script


def test_english_sentences_are_plain_latin():
    for t in [x["en"] for x in prompts.SENTENCES.values()] + [x["en"] for x in prompts.QUESTIONS.values()]:
        assert not scripts_in(t), t


def test_all_extractor_fields_have_labels_and_questions():
    for name in extractor.list_fields():
        assert name in prompts.LABELS and name in prompts.QUESTIONS, name
        for lang in prompts.LANGS:
            assert prompts.LABELS[name][lang].strip() and prompts.QUESTIONS[name][lang].strip()


def test_say_formats_and_falls_back():
    assert prompts.say("confirm", "en", label="age", value="25") == "I heard your age as 25. Is that correct?"
    assert prompts.say("done", "xx") == prompts.say("done", "en")
    assert "Clinic" in prompts.say("greeting", "hi", title="Clinic")
    with pytest.raises(KeyError):
        prompts.say("nonsense", "en")


def test_overrides_win(monkeypatch):
    monkeypatch.setitem(prompts.OVERRIDES, "done", {"en": "All done!"})
    assert prompts.say("done", "en") == "All done!" and prompts.say("done", "hi") != "All done!"


def test_default_question_for_unknown_field_uses_generic_template():
    assert prompts.default_question("ration_card", "en", "ration card") == "Please tell me your ration card."


def test_spoken_value():
    assert prompts.spoken_value("phone", "9876543210", "en") == "9 8 7 6 5 4 3 2 1 0"
    assert prompts.spoken_value("gender", "Female", "ml") == "സ്ത്രീ"
    assert prompts.spoken_value("yes_no", "No", "hi") == "नहीं"
    assert prompts.spoken_value("date_of_birth", "1990-03-05", "kn") == "5 ಮಾರ್ಚ್ 1990"
    assert prompts.spoken_value("name", "Anil Kumar", "en") == "Anil Kumar" and prompts.spoken_value("age", None, "en") == ""


# ====================================================================== forms
def test_form_validation_messages():
    bad = [
        ({}, "id"), ({"id": "Bad Id!", "title": "x", "fields": [{"name": "age"}]}, "id"),
        ({"id": "f", "fields": [{"name": "age"}]}, "title"), ({"id": "f", "title": "x"}, "fields"),
        ({"id": "f", "title": "x", "fields": []}, "fields"), ({"id": "f", "title": "x", "fields": [{}]}, "name"),
        ({"id": "f", "title": "x", "fields": [{"name": "age"}, {"name": "AGE"}]}, "twice"),
        ({"id": "f", "title": "x", "fields": [{"name": "zzz", "kind": "nope"}]}, "kind"),
        ({"id": "f", "title": "x", "fields": [{"name": "age", "question": 5}]}, "question"),
        ({"id": "f", "title": "x", "fields": [{"name": "bad name!!", "kind": "text"}]}, "may only contain"),
        ("not a dict", "object"), ({"id": "f", "title": "x", "fields": "age"}, "fields"),
    ]
    for data, word in bad:
        with pytest.raises(FormError, match=word):
            FormSpec.from_dict(data)


def test_form_normalisation_and_defaults():
    f = FormSpec.from_dict({"id": "F1", "title": " Clinic ", "language": "HI-IN", "fields": ["name", {"name": "DOB"}, {"name": "Mobile", "required": False}]})
    assert f.id == "f1" and f.title == "Clinic" and f.language == "hi" and "hi" in f.languages
    assert [x.name for x in f.fields] == ["name", "date_of_birth", "phone"] and f.fields[2].required is False
    assert f.settings["require_consent"] is True and f.settings["store_audio"] is False


def test_unknown_settings_and_wrong_types_are_ignored():
    f = FormSpec.from_dict({"id": "f", "title": "x", "fields": ["age"], "settings": {"require_consent": "no", "evil": 1, "store_audio": True}})
    assert f.settings["require_consent"] is True and f.settings["store_audio"] is True and "evil" not in f.settings


def test_questions_and_labels_custom_and_default():
    f = FormSpec.from_dict({"id": "f", "title": "x", "fields": [
        {"name": "age", "question": {"en": "Your age please?", "hi": "आपकी उम्र?"}}, {"name": "name"}]})
    assert f.fields[0].question_for("en") == "Your age please?" and f.fields[0].question_for("hi") == "आपकी उम्र?"
    assert f.fields[0].question_for("ml") == "Your age please?"                      # custom text has no ml: use the English one
    assert f.fields[1].question_for("hi") == prompts.QUESTIONS["name"]["hi"]


def test_custom_field_registered_with_extractor():
    try:
        f = FormSpec.from_dict({"id": "f", "title": "x", "fields": [{"name": "Ration Card No", "kind": "text", "label": {"en": "ration card number"}}]})
        assert f.fields[0].name == "ration_card_no" and extractor.resolve_field("ration_card_no").kind == "text"
        assert f.fields[0].label_for("en") == "ration card number"
        assert "ration card number" in f.fields[0].question_for("en")
    finally:
        extractor.FIELDS.pop("ration_card_no", None); extractor._ALIASES.pop("ration_card_no", None)


def test_sensitive_builtin_fields_are_marked():
    f = FormSpec.from_dict({"id": "f", "title": "x", "fields": ["aadhaar", "age", {"name": "ration", "kind": "text", "sensitive": True}]})
    try:
        assert [x.sensitive for x in f.fields] == [True, False, True]
    finally:
        extractor.FIELDS.pop("ration", None); extractor._ALIASES.pop("ration", None)


def test_form_roundtrip_through_dict():
    f = FormSpec.load(os.path.join(HERE, "..", "data", "forms", "clinic_registration.json"))
    g = FormSpec.from_dict(f.to_dict())
    assert [x.name for x in g.fields] == [x.name for x in f.fields] and g.title == f.title


def test_form_store(tmp_path):
    st = FormStore(str(tmp_path / "forms"))
    assert st.list() == []
    st.save({"id": "clinic", "title": "Clinic", "fields": ["name", "age"]})
    st.save({"id": "scheme", "title": "Scheme", "fields": ["name"]})
    assert st.list() == [("clinic", "Clinic"), ("scheme", "Scheme")] and st.get("clinic").fields[1].name == "age"
    (tmp_path / "forms" / "broken.json").write_text("{ nope")
    assert [i for i, _ in st.list()] == ["clinic", "scheme"]                          # a broken file does not hide the others
    with pytest.raises(FormError, match="not valid JSON"):
        FormSpec.load(str(tmp_path / "forms" / "broken.json"))
    for bad in ("../x", "a/b", "", "UPPER"):
        with pytest.raises(FormError):
            st.get(bad)
    with pytest.raises(FormError, match="not found"):
        st.get("missing")
    assert not [f for f in os.listdir(tmp_path / "forms") if f.endswith(".tmp")]


def test_form_store_save_is_validated_and_does_not_write_on_error(tmp_path):
    st = FormStore(str(tmp_path))
    with pytest.raises(FormError):
        st.save({"id": "x", "title": "t", "fields": []})
    assert os.listdir(tmp_path) == []


# ====================================================================== call store
def rec(cid="abcdef0123456789", started="2026-11-13T10:00:00+00:00"):
    return {"call_id": cid, "form_id": "clinic", "form_title": "Clinic", "language": "en", "started_at": started,
            "ended_at": None, "outcome": "in_progress", "progress": [0, 5], "fields": [], "events": []}


def test_store_roundtrip_unicode_and_atomic(tmp_path):
    st = CallStore(str(tmp_path / "calls"))
    r = rec(); r["fields"] = [{"name": "name", "value": "सुनीता"}]
    st.save(r); r["outcome"] = "completed"; st.save(r)
    assert st.load("abcdef0123456789")["outcome"] == "completed"
    assert "सुनीता" in open(tmp_path / "calls" / "abcdef0123456789.json", encoding="utf-8").read()      # readable, not \u escapes
    assert [f for f in os.listdir(tmp_path / "calls") if f.endswith(".tmp")] == []


def test_store_rejects_path_tricks(tmp_path):
    st = CallStore(str(tmp_path))
    for bad in ("../secret", "..", "a/b", "abc", "ZZZZZZZZZZ", "", None, "abcdef01/../../x"):
        with pytest.raises(StoreError):
            st.load(bad)
        with pytest.raises(StoreError):
            st.delete(bad)
    with pytest.raises(StoreError):
        st.save(rec(cid="../evil"))
    with pytest.raises(StoreError, match="not found"):
        st.load("0123456789abcdef")


def test_store_list_newest_first_and_ignores_junk(tmp_path):
    st = CallStore(str(tmp_path))
    st.save(rec("aaaaaaaaaaaaaaaa", "2026-11-13T09:00:00+00:00")); st.save(rec("bbbbbbbbbbbbbbbb", "2026-11-13T11:00:00+00:00"))
    (tmp_path / "junk.json").write_text("not json"); (tmp_path / "notes.txt").write_text("x")
    assert [r["call_id"] for r in st.list()] == ["bbbbbbbbbbbbbbbb", "aaaaaaaaaaaaaaaa"]


def test_store_audio_and_delete_and_purge(tmp_path):
    st = CallStore(str(tmp_path))
    st.save(rec()); p = st.save_audio("abcdef0123456789", 3, b"RIFFxxxx")
    assert p.endswith("turn_003.wav") and os.path.exists(p)
    assert st.delete("abcdef0123456789") is True and not os.path.exists(p) and st.delete("abcdef0123456789") is False
    st.save(rec("cccccccccccccccc")); old = tmp_path / "cccccccccccccccc.json"
    os.utime(old, (1, 1))
    st.save(rec("dddddddddddddddd"))
    assert st.purge_older_than(30) == 1 and [r["call_id"] for r in st.list()] == ["dddddddddddddddd"]
