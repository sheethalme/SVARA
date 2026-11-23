"""Tests for the call controller (callpage/session.py): every path a real call can take, without audio or network."""
import datetime as dt
import json
import os

import pytest

import extractor
from callpage import prompts
from callpage.forms import FormSpec
from callpage.session import CallSession, TranscribeResult, is_noise_text, match_field
from callpage.store import CallStore

TODAY = dt.date(2026, 11, 13)
HERE = os.path.dirname(__file__)


class Clock:
    def __init__(self): self.t = 1_800_000_000.0
    def __call__(self): self.t += 1.0; return self.t


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "dotenv", None)
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "SVARA_LLM_API_KEY", "SVARA_LLM_BASE_URL", "SVARA_EXTRACTOR_LLM"):
        monkeypatch.delenv(k, raising=False)
    extractor.reset_extractor()


def form(fields=None, **settings):
    fields = fields or [{"name": "name"}, {"name": "age"}, {"name": "gender"}, {"name": "phone"},
                        {"name": "blood_group", "required": False}]
    return FormSpec.from_dict({"id": "clinic", "title": "Clinic registration", "language": "en",
                               "settings": settings, "fields": fields})


def session(f=None, lang=None, store=None, **kw):
    return CallSession(f or form(), lang, store=store, clock=Clock(), call_id="abc123def456", today=TODAY,
                       ex=extractor.Extractor(today=TODAY), **kw)


def say(s, text):
    return s.handle_text(text)


def run(s, replies):
    """Start the call and feed one reply per turn; returns all turns."""
    turns = [s.start()]
    for r in replies:
        t = s.handle_no_speech() if r is None else s.handle_text(r)
        turns.append(t)
        if t.done:
            break
    return turns


HAPPY = ["yes", "my name is anil kumar", "yes", "I am twenty five years old", "yes", "male", "yes",
         "nine eight seven six five four three two one zero", "yes", "o negative", "yes", "yes"]


# ====================================================================== the happy path
def test_full_call_happy_path():
    s = session()
    turns = run(s, HAPPY)
    assert turns[-1].done and turns[-1].outcome == "completed" and s.finished
    vals = {f["name"]: f["value"] for f in s.record()["fields"]}
    assert vals == {"name": "Anil Kumar", "age": "25", "gender": "Male", "phone": "9876543210", "blood_group": "O-"}
    assert all(f["status"] == "confirmed" and f["confirmed_at"] for f in s.record()["fields"])
    assert s.record()["consent"] is True and s.record()["outcome"] == "completed"


def test_the_questions_come_from_the_form_in_order():
    s = session()
    turns = run(s, HAPPY)
    qs = [t.say for t in turns if t.phase == "ask" and t.field]
    assert qs[0].endswith("What is your full name?") and "How old are you?" in qs[1]
    assert [t.field for t in turns if t.phase == "ask"][:4] == ["name", "age", "gender", "phone"]


def test_first_turn_asks_consent_and_names_the_form():
    t = session().start()
    assert "Clinic registration" in t.say and "yes or no" in t.say and t.phase == "consent" and t.listen_mode == "yesno"


def test_confirmation_reads_the_value_back():
    s = session()
    run(s, ["yes", "my name is anil kumar"])
    t = s.handle_text("my name is anil kumar") if False else None
    s2 = session(); turns = run(s2, ["yes", "my name is anil kumar"])
    assert turns[-1].say == "I heard your name as Anil Kumar. Is that correct?" and turns[-1].phase == "confirm"
    assert turns[-1].listen_mode == "yesno" and turns[-1].value == "Anil Kumar"


def test_phone_is_read_digit_by_digit():
    s = session()
    turns = run(s, ["yes", "anil", "yes", "30", "yes", "male", "yes", "98765 43210"])
    assert turns[-1].say == "I heard your phone number as 9 8 7 6 5 4 3 2 1 0. Is that correct?"


def test_progress_counts_finished_fields():
    s = session()
    turns = run(s, HAPPY)
    assert turns[0].progress == (0, 5) and turns[-1].progress == (5, 5)
    mid = [t for t in turns if t.phase == "ask" and t.field == "gender"][0]
    assert mid.progress == (2, 5)


# ====================================================================== corrections and repeats
def test_user_corrects_the_value_in_the_confirmation_turn():
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", "I am twenty five", "no, it is twenty six"])
    assert turns[-1].say == "Okay. So your age is 26. Is that correct?" and turns[-1].phase == "confirm"
    t = s.handle_text("yes")
    assert t.phase == "ask" and s.record()["fields"][1]["value"] == "26"


def test_say_again_repeats_the_readback_not_the_question():
    s = session()
    turns = run(s, ["yes", "name is anil", "say again"])
    assert turns[-1].phase == "confirm" and turns[-1].say == "I heard your name as Anil. Is that correct?"


def test_no_say_again_asks_the_question_again():
    s = session()
    turns = run(s, ["yes", "name is anil", "no, say again"])
    assert turns[-1].phase == "ask" and turns[-1].say == "Okay. Please tell me your name again."
    t = s.handle_text("my name is sunil"); assert "Sunil" in t.say


def test_plain_no_asks_again_and_keeps_nothing():
    s = session()
    run(s, ["yes", "name is anil", "no"])
    assert s.record()["fields"][0]["value"] is None and s.record()["fields"][0]["rejections"] == 1


def test_repeated_rejections_give_up_on_a_required_field():
    s = session()
    turns = run(s, ["yes", "name is anil", "no", "name is anil", "no", "name is anil", "no"])
    assert turns[-1].done and turns[-1].outcome == "needs_help"
    assert s.record()["fields"][0]["status"] == "needs_help"


def test_unclear_confirmation_asks_yes_or_no():
    s = session()
    turns = run(s, ["yes", "name is anil", "maybe the weather"])
    assert turns[-1].say.startswith("Please say yes or no.") and "Is that correct?" in turns[-1].say


def test_three_unclear_confirmations_count_as_a_failed_try():
    s = session()
    turns = run(s, ["yes", "name is anil", "banana", "banana", "banana"])
    assert turns[-1].phase == "ask" and "did not catch" in turns[-1].say


# ====================================================================== answers that cannot be read
def test_unreadable_answer_is_asked_again_with_the_reason():
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", "blah blah"])
    assert turns[-1].phase == "ask" and turns[-1].say.startswith("Sorry, I did not catch that.") and "age" in turns[-1].say.lower()


def test_invalid_value_gives_the_extractor_reason():
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", "one hundred and fifty", ])
    assert "between 0 and 120" in turns[-1].say


def test_required_field_failing_three_times_ends_with_needs_help():
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", "blah", "blah", "blah"])
    assert turns[-1].done and turns[-1].outcome == "needs_help" and "person will help" in turns[-1].say
    rec = s.record()
    assert rec["fields"][0]["status"] == "confirmed" and rec["fields"][1]["status"] == "needs_help"      # name is kept


def test_optional_field_is_skipped_after_failures():
    f = form([{"name": "name"}, {"name": "blood_group", "required": False}, {"name": "age"}])
    s = session(f)
    turns = run(s, ["yes", "name is anil", "yes", "blah", "blah", "blah"])
    assert s.record()["fields"][1]["status"] == "skipped" and turns[-1].field == "age"


def test_skip_by_voice_only_for_optional_fields():
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", "skip"])
    assert "needed for the form" in turns[-1].say and turns[-1].field == "age"
    f = form([{"name": "name"}, {"name": "blood_group", "required": False}, {"name": "age"}])
    s2 = session(f); turns2 = run(s2, ["yes", "name is anil", "yes", "I don't know"])
    assert s2.record()["fields"][1]["status"] == "skipped" and turns2[-1].field == "age" and turns2[-1].say.startswith("Okay, let us skip")


@pytest.mark.parametrize("junk", ["Thank you.", "Thanks for watching!", "you", "um", "...", "", "Bye.", "।", "  "])
def test_recogniser_hallucinations_are_treated_as_silence(junk):
    s = session()
    turns = run(s, ["yes", junk])
    assert turns[-1].say.startswith("I could not hear you.") and s.record()["fields"][0]["attempts"] == 0
    assert is_noise_text(junk)


# ====================================================================== silence, errors, hang-up, consent
def test_three_silences_end_the_call_and_keep_the_answers():
    store_dir = None
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", None, None, None])
    assert turns[-1].done and turns[-1].outcome == "timed_out" and "Please call again" in turns[-1].say
    assert s.record()["fields"][0]["value"] == "Anil" and s.record()["outcome"] == "timed_out"


def test_silence_repeats_the_current_question():
    s = session()
    turns = run(s, ["yes", None])
    assert turns[-1].say == "I could not hear you. What is your full name?"
    t = s.handle_text("name is anil"); assert t.phase == "confirm"
    assert s.silence == 0                                           # an answer resets the silence counter


def test_silence_during_confirmation_repeats_the_readback():
    s = session()
    turns = run(s, ["yes", "name is anil", None])
    assert turns[-1].say == "I could not hear you. I heard your name as Anil. Is that correct?"


def test_consent_declined_ends_the_call():
    s = session()
    turns = run(s, ["no"])
    assert turns[-1].done and turns[-1].outcome == "declined" and s.record()["consent"] is False
    assert all(f["value"] is None for f in s.record()["fields"])


def test_unclear_consent_three_times_ends():
    s = session()
    turns = run(s, ["what", "orange", "banana"])
    assert turns[-1].done and turns[-1].outcome == "declined"


def test_forms_without_consent_start_with_the_first_question():
    s = session(form(require_consent=False))
    t = s.start()
    assert t.phase == "ask" and "What is your full name?" in t.say and "Do you agree" not in t.say and s.record()["consent"] is None


def test_hangup_saves_progress():
    s = session()
    run(s, ["yes", "name is anil", "yes"])
    t = s.hangup()
    assert t.done and t.outcome == "hangup" and s.record()["fields"][0]["value"] == "Anil"
    assert s.hangup().say == "" and s.handle_text("hello").done                       # idempotent after the end


def test_transcriber_errors_are_retried_then_end():
    calls = []

    def bad(wav, lang, hint):
        calls.append(hint); return TranscribeResult(error="network error")
    s = session(transcriber=bad)
    s.start()
    t1 = s.handle_audio(b"x"); assert "something went wrong" in t1.say and not t1.done
    s.handle_audio(b"x"); t3 = s.handle_audio(b"x")
    assert t3.done and t3.outcome == "error" and len(calls) == 3


def test_transcriber_exception_does_not_crash_the_call():
    def boom(wav, lang, hint): raise RuntimeError("kaboom")
    s = session(transcriber=boom); s.start()
    assert "something went wrong" in s.handle_audio(b"x").say


def test_handle_audio_uses_the_transcriber_and_passes_a_hint_on_yes_no_turns():
    seen = []

    def tr(wav, lang, hint):
        seen.append((lang, hint)); return TranscribeResult(text="yes" if len(seen) == 1 else "name is anil")
    s = session(transcriber=tr); s.start()
    s.handle_audio(b"1"); s.handle_audio(b"2")
    assert seen[0] == ("en", prompts.CONFIRM_HINT["en"]) and seen[1] == ("en", None)


def test_skipped_audio_counts_as_silence():
    s = session(transcriber=lambda w, l, h: TranscribeResult(skipped="silence")); s.start()
    assert s.handle_audio(b"x").say.startswith("I could not hear you.")


# ====================================================================== review and change
def reach_review():
    s = session()
    turns = run(s, ["yes", "name is anil", "yes", "30", "yes", "male", "yes", "9876543210", "yes", "skip"])
    return s, turns


def test_review_reads_everything_and_skipped_fields():
    s, turns = reach_review()
    r = turns[-1]
    assert r.phase == "review" and r.listen_mode == "yesno"
    assert "name: Anil." in r.say and "age: 30." in r.say and "phone number: 9 8 7 6 5 4 3 2 1 0." in r.say
    assert "blood group: skipped." in r.say and r.say.endswith("Is everything correct?")


def test_review_yes_completes():
    s, _ = reach_review()
    assert s.handle_text("yes").outcome == "completed"


def test_change_a_field_after_the_review():
    s, _ = reach_review()
    t = s.handle_text("no"); assert t.phase == "change" and "Which one" in t.say
    t = s.handle_text("change my phone number"); assert t.phase == "ask" and t.field == "phone"
    t = s.handle_text("nine eight seven six five four three two one one"); assert t.phase == "confirm"
    t = s.handle_text("yes"); assert t.phase == "review" and "phone number: 9 8 7 6 5 4 3 2 1 1." in t.say
    assert s.handle_text("yes").outcome == "completed"
    assert s.record()["fields"][3]["value"] == "9876543211"


def test_change_by_position():
    s, _ = reach_review(); s.handle_text("no")
    assert s.handle_text("the second one").field == "age"


def test_change_not_understood_twice_returns_to_the_review():
    s, _ = reach_review(); s.handle_text("no")
    t1 = s.handle_text("hmm the thing"); assert "for example name" in t1.say and t1.phase == "change"
    t2 = s.handle_text("whatever"); assert t2.phase == "review"


def test_saying_yes_while_choosing_what_to_change_finishes():
    s, _ = reach_review(); s.handle_text("no")
    assert s.handle_text("yes it is fine").outcome == "completed"


def test_unclear_review_reply_asks_again_then_needs_help():
    s, _ = reach_review()
    assert s.handle_text("banana").say.startswith("Please say yes or no.")
    s.handle_text("banana"); assert s.handle_text("banana").outcome == "needs_help"


# ====================================================================== match_field
def _fs(f):
    return session(f).fields


@pytest.mark.parametrize("text,expected", [
    ("change my phone number", 3), ("the age please", 1), ("blood group", 4), ("number two", 1), ("mobile", 3),
    ("I want to change name", 0), ("the third one", 2), ("fone", None), ("nothing matches here", None)])
def test_match_field(text, expected):
    assert match_field(text, _fs(None), "en") == expected


def test_match_field_in_hindi_and_malayalam():
    fs = _fs(None)
    assert match_field("मुझे फ़ोन नंबर बदलना है", fs, "hi") == 3
    assert match_field("പ്രായം", fs, "ml") == 1
    assert match_field("ವಯಸ್ಸು", fs, "kn") == 1


def test_match_field_ambiguous_returns_none():
    f = form([{"name": "phone"}, {"name": "pincode"}])
    assert match_field("number", session(f).fields, "en") is None


# ====================================================================== other languages
def test_hindi_call():
    s = session(lang="hi")
    t = s.start()
    assert "नमस्ते" in t.say and t.lang == "hi"
    turns = run(s, ["हाँ", "मेरा नाम सुनीता शर्मा है"])
    assert turns[-1].say == "मैंने नाम सुनीता शर्मा सुना। क्या यह सही है?"
    t = s.handle_text("सही है"); assert t.say == "आपकी उम्र कितनी है?"
    t = s.handle_text("पच्चीस साल"); assert t.say == "मैंने उम्र 25 सुना। क्या यह सही है?"
    t = s.handle_text("हाँ"); t = s.handle_text("महिला"); assert t.say == "मैंने लिंग महिला सुना। क्या यह सही है?"


def test_hindi_correction_and_say_again():
    s = session(lang="hi"); run(s, ["हाँ", "मेरा नाम अनिल है", "नहीं, फिर से"])
    assert s.phase == "ask"
    run_ = s.handle_text("मेरा नाम सुनील है"); assert "सुनील" in run_.say


def test_malayalam_and_kannada_prompts_are_used():
    assert "ഫോം" in session(lang="ml").start().say and "ಫಾರ್ಮ್" in session(lang="kn").start().say


def test_unknown_language_falls_back_to_english():
    assert session(lang="xx").start().say.startswith("Hello.")


def test_date_of_birth_is_read_with_month_names():
    f = form([{"name": "date_of_birth"}])
    s = session(f, lang="hi"); run(s, ["हाँ", "पाँच मार्च उन्नीस सौ नब्बे"])
    assert "5 मार्च 1990" in s.last_say


def test_gender_is_read_in_the_callers_language():
    f = form([{"name": "gender"}])
    s = session(f, lang="hi"); run(s, ["हाँ", "female"])
    assert "महिला" in s.last_say


# ====================================================================== custom fields
def test_custom_field_with_its_own_question():
    f = FormSpec.from_dict({"id": "scheme", "title": "Scheme", "fields": [
        {"name": "ration_card_no", "kind": "text", "label": {"en": "ration card number"},
         "question": {"en": "What is your ration card number?"}}], "settings": {"require_consent": False}})
    try:
        s = session(f)
        t = s.start()
        assert "What is your ration card number?" in t.say
        t = s.handle_text("it is WB 1234"); assert "ration card number as" in t.say
    finally:
        extractor.FIELDS.pop("ration_card_no", None); extractor._ALIASES.pop("ration_card_no", None)


def test_aliases_in_the_form_resolve_to_builtin_fields():
    f = form([{"name": "DOB"}, {"name": "Mobile"}, {"name": "Blood Type"}])
    assert [x.spec.name for x in session(f).fields] == ["date_of_birth", "phone", "blood_group"]


# ====================================================================== record and privacy
def test_sensitive_fields_are_redacted_in_the_log():
    f = form([{"name": "aadhaar"}, {"name": "name"}])
    s = session(f)
    aad = "234567890124"
    run(s, ["yes", " ".join(aad), "yes"])
    log = json.dumps(s.record()["events"])
    assert aad not in log and "2 3 4 5" not in log and "[redacted]" in log
    assert s.record()["fields"][0]["value"] == aad                       # the form itself has the real value
    assert s.record()["fields"][0]["raw_answers"] == ["[redacted]"]


def test_redaction_can_be_switched_off():
    f = form([{"name": "aadhaar"}], redact_sensitive_log=False)
    s = session(f); run(s, ["yes", "234567890124"])
    assert "234567890124" in json.dumps(s.record()["events"])


def test_record_structure_and_json_serialisable():
    s = session(); run(s, HAPPY)
    rec = s.record()
    assert {"call_id", "form_id", "form_title", "language", "started_at", "ended_at", "outcome", "consent", "fields",
            "events", "settings", "progress"} <= set(rec)
    assert json.dumps(rec, ensure_ascii=False)
    assert rec["events"][0]["role"] == "system" and any(e["role"] == "user" for e in rec["events"])
    assert rec["started_at"].endswith("+00:00")


def test_session_saves_after_every_turn(tmp_path):
    store = CallStore(str(tmp_path))
    s = session(store=store)
    s.start(); assert store.load("abc123def456")["outcome"] == "in_progress"
    s.handle_text("yes"); s.handle_text("name is anil"); s.handle_text("yes")
    saved = store.load("abc123def456")
    assert saved["fields"][0]["value"] == "Anil" and saved["progress"] == [1, 5]
    s.hangup(); assert store.load("abc123def456")["outcome"] == "hangup"


def test_a_failing_disk_does_not_stop_the_call():
    class Broken:
        def save(self, r): raise OSError("disk full")
    s = session(store=Broken()); s.start()
    assert s.handle_text("yes").phase == "ask"


def test_sample_form_file_loads_and_runs():
    f = FormSpec.load(os.path.join(HERE, "..", "data", "forms", "clinic_registration.json"))
    assert [x.name for x in f.fields] == ["name", "age", "gender", "phone", "blood_group"]
    s = session(f); turns = run(s, HAPPY)
    assert turns[-1].outcome == "completed"


@pytest.mark.parametrize("text", ["5", "हाँ", "नहीं", "ശരി", "ಹೌದು", "a", "9", "७"])
def test_short_and_indic_answers_are_not_mistaken_for_noise(text):
    """Regression: a regex word-character filter threw away Hindi / Malayalam / Kannada words and one-digit answers."""
    assert not is_noise_text(text)


def test_single_digit_age_is_accepted():
    s = session(); run(s, ["yes", "name is anil", "yes"])
    assert "age as 5" in s.handle_text("5").say
