"""Guided, voice-driven form filling (builds on 2A speech-to-text and 2B field extraction).

* An admin saves a *form*: a title, an optional uploaded file (PDF / image / Word) and the list of fields the
  user must fill, each with the question to ask.
* ``FormFlow`` then runs the dialogue one field at a time:

      ask the question -> user answers -> read the value back (with spelling for names)
          -> "yes"  : save it, go to the next field
          -> "no, it is ..." / spelled letters : take the corrected value, read it back again
          -> "no" / "say again"                : ask the same question again

  Everything the user says goes through ``FormFlow.handle(text)``, so the flow itself needs no audio and can
  be tested with plain strings.  ``transcribe()`` turns a recording into that text (1A + 1B + 2A).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

import extractor

ROOT = os.path.dirname(os.path.abspath(__file__))
FORMS_DIR = os.environ.get("SVARA_FORMS_DIR") or os.path.join(ROOT, "forms")
ALLOWED_UPLOADS = {".pdf", ".png", ".jpg", ".jpeg", ".docx"}
LANGUAGES = {"en": "English", "hi": "Hindi", "ml": "Malayalam", "kn": "Kannada"}
TTS_LANG = {"en": "en-IN", "hi": "hi-IN", "ml": "ml-IN", "kn": "kn-IN"}   # browser speech-synthesis codes

DEFAULT_QUESTIONS = {
    "name": "Please tell me your full name.", "age": "How old are you?", "gender": "What is your gender?",
    "phone": "Please say your mobile number.", "date_of_birth": "What is your date of birth?",
    "date": "What is the date?", "address": "Please tell me your address.", "pincode": "What is your PIN code?",
    "aadhaar": "Please say your Aadhaar number.", "blood_group": "What is your blood group?",
    "email": "Please say your e-mail address.", "complaint": "What is the problem, or the reason for your visit?",
    "occupation": "What is your occupation?", "yes_no": "Please answer yes or no.", "amount": "What is the amount?",
}


# ------------------------------------------------------------------------------------------- form storage
def default_question(name: str, label: str = "") -> str:
    return DEFAULT_QUESTIONS.get(name) or f"Please tell me your {label or name.replace('_', ' ')}."


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "field"


def make_field(name: str, question: str = "", required: bool = True, custom: bool = False, kind: str = "text",
               label: str = "") -> Dict[str, Any]:
    """One field of a form.  Built-in fields use ``name`` from ``extractor.list_fields()``."""
    if not custom:
        spec = extractor.resolve_field(name)
        name, kind, label = spec.name, spec.kind, spec.label or spec.name
    return {"name": slug(name), "label": label or name.replace("_", " "), "kind": kind, "custom": custom,
            "required": bool(required), "question": question.strip() or default_question(slug(name), label)}


def register_form_fields(form: Dict[str, Any]) -> None:
    """Make the form's custom fields known to the extractor (they are not saved inside extractor.py)."""
    for f in form["fields"]:
        if f.get("custom"):
            extractor.register_field(extractor.FieldSpec(
                name=f["name"], kind=f["kind"], label=f["label"],
                hint=f"The {f['label']} exactly as the user said it.", llm="fallback"))


def _form_dir(form_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{12}", form_id or ""):
        raise ValueError("bad form id")
    return os.path.join(FORMS_DIR, form_id)


def save_form(title: str, fields: List[Dict[str, Any]], file_bytes: Optional[bytes] = None,
              file_name: Optional[str] = None) -> Dict[str, Any]:
    if not title.strip():
        raise ValueError("Give the form a title.")
    if not fields:
        raise ValueError("Choose at least one field.")
    form_id = uuid.uuid4().hex[:12]
    d = _form_dir(form_id)
    os.makedirs(d)
    original = None
    if file_bytes and file_name:
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ALLOWED_UPLOADS:
            shutil.rmtree(d, ignore_errors=True)
            raise ValueError(f"File type {ext!r} is not allowed. Use: {', '.join(sorted(ALLOWED_UPLOADS))}")
        original = "original" + ext
        with open(os.path.join(d, original), "wb") as fh:
            fh.write(file_bytes)
    form = {"id": form_id, "title": title.strip(), "created": time.strftime("%Y-%m-%d %H:%M"),
            "original_file": original, "original_name": file_name if original else None, "fields": fields}
    with open(os.path.join(d, "form.json"), "w", encoding="utf-8") as fh:
        json.dump(form, fh, ensure_ascii=False, indent=2)
    return form


def load_form(form_id: str) -> Dict[str, Any]:
    with open(os.path.join(_form_dir(form_id), "form.json"), encoding="utf-8") as fh:
        form = json.load(fh)
    register_form_fields(form)
    return form


def list_forms() -> List[Dict[str, Any]]:
    out = []
    if os.path.isdir(FORMS_DIR):
        for n in os.listdir(FORMS_DIR):
            try:
                out.append(load_form(n))
            except (OSError, ValueError, KeyError):
                continue
    return sorted(out, key=lambda f: f["created"], reverse=True)


def delete_form(form_id: str) -> None:
    shutil.rmtree(_form_dir(form_id), ignore_errors=True)


def original_path(form: Dict[str, Any]) -> Optional[str]:
    return os.path.join(_form_dir(form["id"]), form["original_file"]) if form.get("original_file") else None


def save_submission(form: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    d = os.path.join(_form_dir(form["id"]), "submissions")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, time.strftime("%Y%m%d-%H%M%S") + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"form": form["title"], "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "answers": rows},
                  fh, ensure_ascii=False, indent=2)
    return path


# ------------------------------------------------------------------------------------- speech clean-up
# Whisper writes spelled-out letters as "A D H I R A", "A-D-H-I-R-A" or "A. D. H. I. R. A."; join them into a word.
_HYPHEN_RUN = re.compile(r"(?<![A-Za-z'’])(?:[A-Za-z][\-.]){2,}[A-Za-z]\.?(?![A-Za-z])")
_SPACE_RUN = re.compile(r"(?<![A-Za-z'’])(?:[A-Za-z][\s.,\-]+){2,}[A-Za-z](?![A-Za-z])")
_LEADS = re.compile(r"\b(?:it'?s|it is|it should be|should be|change (?:it )?to|spelled|spelt|spelling is|"
                    r"the spelling is|the correct spelling is|the correct one is|correct one is)\b[:,]?\s*", re.I)


def join_spelled_letters(text: str) -> str:
    """'no it is A D H I R A' -> 'no it is Adhira'.  Needs at least three single letters in a row."""
    def _join(m: "re.Match[str]") -> str:
        return "".join(re.findall(r"[A-Za-z]", m.group(0))).capitalize()
    return _SPACE_RUN.sub(_join, _HYPHEN_RUN.sub(_join, text))


def clean_correction(text: str) -> str:
    return re.sub(r"\s+", " ", _LEADS.sub("", join_spelled_letters(text))).strip()


def spell_out(value: str) -> str:
    """'Adira Mohan' -> 'A, D, I, R, A. M, O, H, A, N' (each letter said separately by text-to-speech)."""
    words = [w for w in re.split(r"\s+", value.strip()) if w]
    return ". ".join(", ".join(c.upper() for c in w if c.isalnum()) for w in words)


_EMAIL_SAY = {"@": "at", ".": "dot", "_": "underscore", "-": "dash", "+": "plus"}


def spell_email(value: str) -> str:
    """'ab04@gmail.com' -> 'a, b, 0, 4, at, g, m, a, i, l, dot, c, o, m' so the user can check every letter."""
    return ", ".join(_EMAIL_SAY.get(c, c) for c in value)


# ---------------------------------------------------------------------------------------------- dialogue
class FormFlow:
    """One user filling one form.  ``say`` is always what the system says next (shown and spoken)."""

    def __init__(self, form: Dict[str, Any], language: str = "en"):
        register_form_fields(form)
        self.form, self.language = form, language if language in LANGUAGES else "en"
        self.idx, self.phase = 0, "ask"           # phase: ask | confirm | done
        self.cand: Optional[Any] = None           # the ExtractionResult waiting for confirmation
        self.answers: Dict[str, Dict[str, Any]] = {}
        self.fails = 0
        self.turn = 0                              # increases on every step, so the UI can reset its widgets
        self.heard = ""                            # what the user just said (for display)
        self.notes: List[str] = []                 # issues to show, e.g. "I heard 9 digits"
        self.say = self.field["question"] if form["fields"] else "This form has no fields."
        if not form["fields"]:
            self.phase = "done"

    # ----------------------------------------------------------------------------------------- helpers
    @property
    def field(self) -> Dict[str, Any]:
        return self.form["fields"][min(self.idx, len(self.form["fields"]) - 1)]

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def progress(self) -> float:
        return min(self.idx, len(self.form["fields"])) / max(1, len(self.form["fields"]))

    def _readback(self, prefix: str = "") -> str:
        f, c = self.field, self.cand
        spoken = extractor.spoken_form(f["name"], c.value, self.language)
        text = f"{prefix}I heard your {f['label']} as {c.display or spoken}."
        if f["kind"] == "name":
            text += f" Spelled: {spell_out(c.display or c.value)}."
        elif f["kind"] == "email":
            text = f"{prefix}I heard your {f['label']} as {spoken}. Spelled: {spell_email(c.value)}."
        elif f["kind"] in ("phone", "pincode", "aadhaar"):
            text = f"{prefix}I heard your {f['label']} as {spoken}."
        return text + " Is that correct?"

    def _advance(self) -> None:
        self.idx += 1
        self.phase, self.cand, self.fails = "ask", None, 0
        if self.idx >= len(self.form["fields"]):
            self.phase, self.say = "done", "All done. Thank you."
        else:
            self.say = self.field["question"]

    # ------------------------------------------------------------------------------------------ steps
    def handle(self, text: Optional[str]) -> None:
        """Feed one spoken reply (already transcribed).  Moves the dialogue on."""
        if self.done:
            return
        self.turn += 1
        self.notes = []
        text = (text or "").strip()
        self.heard = text
        if not text:
            self.say = "I did not hear anything. " + (self._readback() if self.phase == "confirm" else self.field["question"])
            return
        (self._confirm if self.phase == "confirm" else self._answer)(text)

    def skip(self) -> bool:
        """Skip the current field (only optional ones)."""
        if self.done or self.field.get("required", True):
            return False
        self.turn += 1
        self.heard, self.notes = "", []
        self._advance()
        return True

    def _answer(self, text: str) -> None:
        f = self.field
        if f["kind"] == "name":
            text = join_spelled_letters(text)
        r = extractor.extract_field_full(text, f["name"], language=self.language)
        if r.value is None:
            self.fails += 1
            self.notes = list(r.issues)
            self.say = f"Sorry, I could not find your {f['label']}. " + f["question"]
            return
        self.cand, self.phase, self.notes = r, "confirm", list(r.issues)
        self.say = self._readback()

    def _confirm(self, text: str) -> None:
        f = self.field
        o = extractor.handle_confirmation(clean_correction(text), f["name"], self.language)
        if o.action == "accept":
            c = self.cand
            self.answers[f["name"]] = {"field": f["name"], "label": f["label"], "value": c.value,
                                       "display": c.display or c.value, "status": c.status}
            self._advance()
        elif o.action == "update" and o.extraction is not None:
            self.cand, self.notes = o.extraction, list(o.extraction.issues)
            self.say = self._readback("Okay. ")
        elif o.action == "repeat_readback":
            self.say = self._readback()
        elif o.action == "ask_again":
            self.say = "Sorry, I did not understand. Please say yes if it is correct, or say no and then the correct value."
        else:                                       # reask_field
            self.phase, self.cand = "ask", None
            self.say = "Okay, let's try again. " + f["question"]

    def rows(self) -> List[Dict[str, Any]]:
        """The answers so far, in form order."""
        return [self.answers[f["name"]] for f in self.form["fields"] if f["name"] in self.answers]


# ------------------------------------------------------------------------------------- audio -> text
def transcribe(source: Any, language: Optional[str] = None) -> str:
    """Recording -> text using 1A (load), 1B (clean) and 2A (speech-to-text).  Returns '' if no speech."""
    import audio_io
    import preprocess as pp
    import stt
    x, sr, _ = audio_io.load_audio_with_report(source)
    res = pp.clean_audio_with_stages(x, sr, pre_emphasis=0.97, bandpass=True, denoise="spectral_subtraction", vad=True)
    if not res.report.speech_detected:
        return ""
    return (stt.transcribe_full(res.signal, language=language, engine="auto").text or "").strip()
