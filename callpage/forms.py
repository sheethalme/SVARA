"""
forms.py -- the form that the voice call asks, loaded from JSON (written by the Admin "set up a form" page)
=========================================================================================================

A form file (``data/forms/<id>.json``)::

    {
      "id": "clinic_registration",
      "title": "Clinic registration",
      "language": "en",                         # default language of the call (en | hi | ml | kn)
      "languages": ["en", "hi", "ml", "kn"],    # optional: languages the caller may choose
      "settings": {"require_consent": true, "store_audio": false},
      "fields": [
        {"name": "name",  "required": true},                         # built-in field: question/label come from prompts.py
        {"name": "age",   "question": {"en": "How old are you?"}},   # optional custom question per language
        {"name": "blood_group", "required": false},
        {"name": "ration_card_no", "kind": "text",                    # a field the admin added: needs a "kind"
         "label": {"en": "ration card number"}, "question": "What is your ration card number?"}
      ]
    }

``name`` may be any name or alias known to ``extractor`` ("dob", "mobile", "blood type" ...).  Unknown names are
registered as new fields of the given ``kind`` (default ``text``); the kinds are those of ``extractor.RULES``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import extractor
from callpage import prompts

__all__ = ["FormError", "FormField", "FormSpec", "FormStore", "slugify", "build_form_dict"]

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SETTINGS = {"require_consent": True, "store_audio": False, "max_attempts": 3, "redact_sensitive_log": True}


class FormError(ValueError):
    """The form definition is invalid (the message says what to fix)."""


def _text_map(value: Any, what: str) -> Dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        return {"en": value}
    if isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        return {prompts.norm_lang(k) if k.split("-")[0] in prompts.LANGS else k: v for k, v in value.items() if v.strip()}
    raise FormError(f"{what} must be a text or an object like {{\"en\": \"...\", \"hi\": \"...\"}}")


@dataclass
class FormField:
    name: str                         # canonical extractor field name
    kind: str                         # extractor kind (rules used to read the answer)
    required: bool = True
    sensitive: bool = False
    label: Dict[str, str] = field(default_factory=dict)
    question: Dict[str, str] = field(default_factory=dict)

    def label_for(self, lang: str) -> str:
        return prompts.field_label(self.name, lang, self.label or None)

    def question_for(self, lang: str) -> str:
        lang = prompts.norm_lang(lang)
        if self.question:
            return self.question.get(lang) or self.question.get("en") or next(iter(self.question.values()))
        return prompts.default_question(self.name, lang, self.label_for(lang))


@dataclass
class FormSpec:
    id: str
    title: str
    language: str = "en"
    languages: List[str] = field(default_factory=lambda: list(prompts.LANGS))
    fields: List[FormField] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=lambda: dict(_SETTINGS))

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormSpec":
        if not isinstance(d, dict):
            raise FormError("a form must be a JSON object")
        fid = str(d.get("id", "")).strip().lower()
        if not _ID.match(fid):
            raise FormError("form 'id' must be 1-64 characters: lower-case letters, digits, '_' or '-'")
        title = str(d.get("title") or "").strip()
        if not title:
            raise FormError("form 'title' is missing")
        lang = prompts.norm_lang(d.get("language"))
        languages = [prompts.norm_lang(x) for x in (d.get("languages") or prompts.LANGS)]
        languages = list(dict.fromkeys(languages)) or list(prompts.LANGS)
        if lang not in languages:
            languages.insert(0, lang)
        raw_fields = d.get("fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            raise FormError("a form needs at least one entry in 'fields'")
        fields: List[FormField] = []
        seen = set()
        for i, f in enumerate(raw_fields, 1):
            if isinstance(f, str):
                f = {"name": f}
            if not isinstance(f, dict) or not str(f.get("name", "")).strip():
                raise FormError(f"field {i}: 'name' is missing")
            ff = cls._field(f, i)
            if ff.name in seen:
                raise FormError(f"field '{ff.name}' appears twice")
            seen.add(ff.name)
            fields.append(ff)
        settings = dict(_SETTINGS)
        extra = d.get("settings") or {}
        if not isinstance(extra, dict):
            raise FormError("'settings' must be an object")
        for k, v in extra.items():
            if k in _SETTINGS and type(v) is type(_SETTINGS[k]):
                settings[k] = v
        return cls(fid, title, lang, languages, fields, settings)

    @staticmethod
    def _field(f: Dict[str, Any], i: int) -> FormField:
        name = str(f["name"]).strip()
        label = _text_map(f.get("label"), f"field {i} 'label'")
        question = _text_map(f.get("question"), f"field {i} 'question'")
        try:
            spec = extractor.resolve_field(name)
            canonical, kind, sensitive = spec.name, spec.kind, spec.sensitive
        except extractor.UnknownFieldError:
            kind = str(f.get("kind") or "text").strip()
            if kind not in extractor.RULES:
                raise FormError(f"field '{name}': unknown kind '{kind}' (choose one of {', '.join(sorted(extractor.RULES))})")
            canonical = re.sub(r"[\s\-]+", "_", name.lower())
            if not re.match(r"^[a-z0-9_]{1,48}$", canonical):
                raise FormError(f"field name '{name}' may only contain letters, digits and '_'")
            sensitive = bool(f.get("sensitive", False))
            extractor.register_field(extractor.FieldSpec(
                canonical, kind, (label.get("en") or canonical.replace("_", " ")), (),
                f"The {label.get('en') or canonical.replace('_', ' ')} from the speaker's answer.",
                "fallback", sensitive))
        return FormField(canonical, kind, bool(f.get("required", True)), bool(f.get("sensitive", sensitive) or sensitive),
                         label, question)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "language": self.language, "languages": self.languages,
                "settings": self.settings,
                "fields": [{"name": f.name, "kind": f.kind, "required": f.required, "sensitive": f.sensitive,
                            **({"label": f.label} if f.label else {}), **({"question": f.question} if f.question else {})}
                           for f in self.fields]}

    @classmethod
    def load(cls, path: str) -> "FormSpec":
        try:
            with open(path, encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except json.JSONDecodeError as exc:
            raise FormError(f"{os.path.basename(path)} is not valid JSON ({exc.msg} at line {exc.lineno})") from None

    def field_by_name(self, name: str) -> Optional[FormField]:
        return next((f for f in self.fields if f.name == name), None)


class FormStore:
    """A folder of form JSON files (the Admin page writes here, the call page reads)."""

    def __init__(self, directory: str):
        self.dir = directory

    def _path(self, form_id: str) -> str:
        if not _ID.match(form_id or ""):
            raise FormError("invalid form id")
        return os.path.join(self.dir, form_id + ".json")

    def list(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if not os.path.isdir(self.dir):
            return out
        for fn in sorted(os.listdir(self.dir)):
            if fn.endswith(".json"):
                try:
                    f = FormSpec.load(os.path.join(self.dir, fn))
                    out.append((f.id, f.title))
                except (FormError, OSError):
                    continue                              # a broken file must not hide the others
        return out

    def get(self, form_id: str) -> FormSpec:
        p = self._path(form_id)
        if not os.path.exists(p):
            raise FormError(f"form '{form_id}' not found")
        return FormSpec.load(p)

    def save(self, data: Dict[str, Any]) -> FormSpec:
        spec = FormSpec.from_dict(data)                  # validates first
        os.makedirs(self.dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({**data, "id": spec.id}, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path(spec.id))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return spec


def slugify(title: str) -> str:
    """"Clinic registration (2026)" -> "clinic_registration_2026" (a valid form id)."""
    s = re.sub(r"[^a-z0-9]+", "_", (title or "").strip().lower()).strip("_")[:64].strip("_")
    return s if s and s[0].isalnum() else "form"


def build_form_dict(title: str, fields: List[Any], language: str = "en", languages: Optional[List[str]] = None,
                    settings: Optional[Dict[str, Any]] = None, form_id: Optional[str] = None) -> Dict[str, Any]:
    """Make the JSON for a form from what the Admin page collected.

    ``fields`` is a list of field names ("age"), or dicts like {"name": "age", "required": False, "question": "...",
    "label": "...", "kind": "text"}.  Validated by ``FormStore.save`` (raises ``FormError`` with a readable message).
    Example::

        FormStore("data/forms").save(build_form_dict("Clinic registration", ["name", "age", {"name": "phone"}]))
    """
    d: Dict[str, Any] = {"id": form_id or slugify(title), "title": title, "language": language,
                         "fields": [f if isinstance(f, dict) else {"name": f} for f in fields]}
    if languages:
        d["languages"] = languages
    if settings:
        d["settings"] = settings
    return d
