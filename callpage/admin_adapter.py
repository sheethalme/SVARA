"""Convert the existing SVARA admin form format into call-page form JSON."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

import extractor
from callpage.forms import FormError, FormStore, build_form_dict, slugify


def _language_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k).split("-")[0]: str(v).strip() for k, v in value.items() if str(v).strip()}
    return value


def convert_admin_form(admin_form: Dict[str, Any]) -> Dict[str, Any]:
    """Pure conversion.  It does not save or register fields."""
    if not isinstance(admin_form, dict):
        raise FormError("The saved admin form must be a JSON object.")
    title = str(admin_form.get("title") or "").strip()
    if not title:
        raise FormError("Give the form a title.")
    raw_fields = admin_form.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise FormError("Choose at least one field.")

    language = str(admin_form.get("language") or "en").strip().lower().split("-")[0]
    languages = admin_form.get("languages")
    if not isinstance(languages, list):
        languages = [language]
    languages = [str(x).strip().lower().split("-")[0] for x in languages if str(x).strip()]

    fields: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_fields, 1):
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            raise FormError(f"Field {index} is invalid.")
        raw_name = str(raw.get("name") or "").strip()
        label_value = raw.get("label")
        if isinstance(label_value, dict):
            raw_label = str(label_value.get("en") or next(iter(label_value.values()), raw_name)).strip()
        else:
            raw_label = str(label_value or raw_name).strip()
        if not raw_name:
            raise FormError(f"Field {index}: 'name' is missing.")

        is_custom = bool(raw.get("custom"))
        if not is_custom:
            try:
                spec = extractor.resolve_field(raw_name)
                is_custom = False
            except extractor.UnknownFieldError:
                is_custom = True

        if is_custom:
            name = re.sub(r"[^a-z0-9]+", "_", raw_label.lower()).strip("_") or slugify(raw_name)
            kind = str(raw.get("kind") or "text").strip() or "text"
            call_field: Dict[str, Any] = {
                "name": name,
                "kind": kind,
                "required": bool(raw.get("required", True)),
                "label": {"en": raw_label},
                "question": _language_values(raw.get("question") or raw.get("Question asked aloud")),
            }
        else:
            name = spec.name
            question = _language_values(raw.get("question") or raw.get("Question asked aloud"))
            call_field = {"name": name, "required": bool(raw.get("required", True))}
            if question:
                call_field["question"] = question

        fields.append(call_field)

    form_id = str(admin_form.get("id") or "").strip().lower() or slugify(title)
    form_id = re.sub(r"[^a-z0-9_-]+", "_", form_id).strip("_")[:64]
    if not form_id:
        form_id = slugify(title)

    settings = admin_form.get("settings")
    if not isinstance(settings, dict):
        settings = None
    return build_form_dict(
        title,
        fields,
        language=language,
        languages=languages,
        settings=settings,
        form_id=form_id,
    )


def adapt_admin_form(admin_form: Dict[str, Any]) -> Dict[str, Any]:
    """Short alias for callers that prefer an adapter name."""
    return convert_admin_form(admin_form)


def save_admin_form(admin_form: Dict[str, Any], forms_dir: str = "data/forms"):
    data = convert_admin_form(admin_form)
    return FormStore(forms_dir).save(data)


def _existing_forms(source_dir: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    if source_dir is None:
        import formflow as ff
        return ff.list_forms()

    out: List[Dict[str, Any]] = []
    if os.path.isdir(source_dir):
        for name in sorted(os.listdir(source_dir)):
            path = os.path.join(source_dir, name)
            form_json = os.path.join(path, "form.json") if os.path.isdir(path) else path
            if os.path.isfile(form_json) and form_json.endswith(".json"):
                try:
                    import json
                    with open(form_json, encoding="utf-8") as fh:
                        value = json.load(fh)
                    if isinstance(value, dict):
                        out.append(value)
                except (OSError, ValueError):
                    continue
    return out


def sync_all_forms(forms_dir: str = "data/forms", source_dir: Optional[str] = None) -> List[Any]:
    """Export all existing admin forms to the call-page JSON folder."""
    store = FormStore(forms_dir)
    saved = []
    for admin_form in _existing_forms(source_dir):
        saved.append(store.save(convert_admin_form(admin_form)))
    return saved
