import pytest
import formflow as ff


@pytest.fixture
def form(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "FORMS_DIR", str(tmp_path))
    fields = [ff.make_field("name"), ff.make_field("age"), ff.make_field("phone", required=False),
              ff.make_field("Ward Number", custom=True, kind="number", label="ward number")]
    return ff.save_form("Clinic form", fields, b"%PDF-1.4", "clinic.pdf")


def test_save_and_load_roundtrip(form):
    f = ff.load_form(form["id"])
    assert f["title"] == "Clinic form" and [x["name"] for x in f["fields"]] == ["name", "age", "phone", "ward_number"]
    assert ff.original_path(f).endswith("original.pdf")
    assert [x["id"] for x in ff.list_forms()] == [form["id"]]


def test_rejects_bad_upload_and_empty_form(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "FORMS_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        ff.save_form("x", [ff.make_field("name")], b"MZ", "evil.exe")
    with pytest.raises(ValueError):
        ff.save_form("x", [])
    with pytest.raises(ValueError):
        ff.load_form("../../etc")


def test_happy_path_all_fields(form):
    fl = ff.FormFlow(form)
    assert "name" in fl.say.lower()
    fl.handle("my name is Adira Mohan");  assert fl.phase == "confirm" and "Adira Mohan" in fl.say and "A, D, I, R, A" in fl.say
    fl.handle("yes");                       assert fl.field["name"] == "age" and fl.phase == "ask"
    fl.handle("I am twenty two years old"); assert "22" in fl.say
    fl.handle("correct");                   assert fl.field["name"] == "phone"
    fl.handle("nine four eight one four four one five eight eight"); assert fl.phase == "confirm"
    fl.handle("yes");                       assert fl.field["name"] == "ward_number"
    fl.handle("twelve"); fl.handle("yes")
    assert fl.done and fl.progress == 1.0
    assert {r["field"]: r["value"] for r in fl.rows()} == {
        "name": "Adira Mohan", "age": "22", "phone": "9481441588", "ward_number": "12"}


def test_spoken_correction_replaces_value_then_moves_on(form):
    fl = ff.FormFlow(form)
    fl.handle("my name is Adira Mohan")
    fl.handle("no my name is Athira Mohan")
    assert fl.phase == "confirm" and "Athira Mohan" in fl.say          # read back again, not yet saved
    assert not fl.answers
    fl.handle("yes")
    assert fl.answers["name"]["value"] == "Athira Mohan" and fl.field["name"] == "age"


def test_correction_by_spelling_letters(form):
    fl = ff.FormFlow(form)
    fl.handle("my name is Adira")
    fl.handle("no it's A T H I R A")
    assert "Athira" in fl.say and fl.phase == "confirm"
    assert ff.join_spelled_letters("A-D-H-I-R-A M-O-H-A-N") == "Adhira Mohan"
    assert ff.join_spelled_letters("I am 22 years old") == "I am 22 years old"


def test_plain_no_asks_the_question_again(form):
    fl = ff.FormFlow(form)
    fl.handle("my name is Adira Mohan")
    fl.handle("no")
    assert fl.phase == "ask" and fl.field["name"] == "name" and "try again" in fl.say


def test_not_found_and_silence_repeat_the_question(form):
    fl = ff.FormFlow(form)
    fl.handle("blah blah blah the weather is nice today in the city of kochi really")
    assert fl.phase == "ask" and fl.field["name"] == "name"
    fl.handle("")
    assert "did not hear" in fl.say and fl.field["name"] == "name"


def test_unclear_confirmation_and_repeat(form):
    fl = ff.FormFlow(form)
    fl.handle("my name is Adira Mohan")
    fl.handle("say that again")
    assert fl.phase == "confirm" and "Adira Mohan" in fl.say


def test_skip_only_optional(form):
    fl = ff.FormFlow(form)
    assert fl.skip() is False
    fl.handle("my name is Adira Mohan"); fl.handle("yes")
    fl.handle("twenty two"); fl.handle("yes")
    assert fl.field["name"] == "phone" and fl.skip() is True and fl.field["name"] == "ward_number"
