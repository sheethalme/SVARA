"""Pure data helpers for the Form Responses admin page."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import io
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import extractor

from callpage.export import to_csv as _export_csv
from callpage.forms import FormSpec, FormStore
from callpage.store import CallStore


PARTIAL_OUTCOMES = ("declined", "timed_out", "hangup", "needs_help", "error")


def mask_sensitive_value(field_name: str, value: Any, sensitive: bool) -> Any:
    if value is None or value == "":
        return value
    if sensitive:
        text = str(value)
        return "*" * max(0, len(text) - 2) + text[-2:]
    return value


def group_calls(store: CallStore, form_store: FormStore) -> Dict[str, Dict[str, Any]]:
    """Group lightweight call summaries by form. Includes forms whose JSON is gone."""
    existing = dict(form_store.list())
    specs: Dict[str, FormSpec] = {}
    for form_id in existing:
        try:
            specs[form_id] = form_store.get(form_id)
        except Exception:
            continue

    grouped: Dict[str, Dict[str, Any]] = {}
    for summary in store.list():
        fid = str(summary.get("form_id") or "unknown")
        if fid not in grouped:
            spec = specs.get(fid)
            title = spec.title if spec else str(summary.get("form_title") or fid)
            grouped[fid] = {
                "form_id": fid,
                "title": title if spec else f"{title} (form removed)",
                "removed": spec is None,
                "spec": spec,
                "calls": [],
            }
        grouped[fid]["calls"].append(summary)
    for fid in existing:
        if fid not in grouped:
            spec = specs[fid]
            grouped[fid] = {"form_id": fid, "title": spec.title, "removed": False,
                            "spec": spec, "calls": []}
    return dict(sorted(grouped.items(), key=lambda item: (item[1]["removed"], item[1]["title"].lower())))


def list_form_groups(store: CallStore, form_store: FormStore) -> Dict[str, Dict[str, Any]]:
    return group_calls(store, form_store)


def initial_form_id(groups: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Choose the form with the newest call by default."""
    newest: Optional[Tuple[str, str]] = None
    for form_id, group in groups.items():
        for call in group.get("calls", []):
            stamp = str(call.get("started_at") or "")
            if newest is None or stamp > newest[1]:
                newest = (form_id, stamp)
    if newest is not None:
        return newest[0]
    for form_id in groups:
        return form_id
    return None


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    calls = list(records)
    completed = sum(1 for r in calls if r.get("outcome") == "completed")
    needs_help = sum(1 for r in calls if r.get("outcome") == "needs_help")
    timed_out_hangup_declined = sum(
        1 for r in calls if r.get("outcome") in ("timed_out", "hangup", "declined")
    )
    partial = len(calls) - completed
    return {
        "total": len(calls),
        "completed": completed,
        "needs_help": needs_help,
        "timed_out_hangup_declined": timed_out_hangup_declined,
        "partial": partial,
        "completion_rate": (completed / len(calls) * 100.0) if calls else 0.0,
    }


def form_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return summarize(records)


def _date_value(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        try:
            return dt.datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None


def filter_records(records: Iterable[Dict[str, Any]], outcome: str = "Completed",
                   start_date: Any = None, end_date: Any = None, language: str = "All",
                   search: str = "") -> List[Dict[str, Any]]:
    start = _date_value(start_date)
    end = _date_value(end_date)
    if end is not None:
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    language = str(language or "All").strip()
    needle = str(search or "").strip().lower()

    out = []
    for r in records:
        if outcome and str(outcome).lower() not in ("all", "*"):
            if str(r.get("outcome") or "").lower() != str(outcome).lower():
                continue
        stamp = _date_value(r.get("started_at"))
        if start is not None and (stamp is None or stamp < start):
            continue
        if end is not None and (stamp is None or stamp > end):
            continue
        if language.lower() not in ("all", "*", ""):
            if str(r.get("language") or "").lower() != language.lower():
                continue
        if needle:
            values = [str(r.get(k) or "") for k in ("call_id", "form_title", "language", "outcome")]
            for correction in r.get("corrections") or []:
                values.extend(str(correction.get(k) or "") for k in ("field", "old", "new"))
            for field in r.get("fields") or []:
                values.extend(str(field.get(k) or "") for k in ("name", "label", "value", "display", "status"))
            if not any(needle in value.lower() for value in values):
                continue
        out.append(r)
    return sorted(out, key=lambda r: r.get("started_at") or "", reverse=True)


def latest_correction(record: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
    found = None
    for correction in record.get("corrections") or []:
        if correction.get("field") == field_name:
            found = correction
    return found


def final_field_value(record: Dict[str, Any], field_name: str) -> Tuple[Any, bool]:
    correction = latest_correction(record, field_name)
    if correction:
        return correction.get("new"), True
    for field in record.get("fields") or []:
        if field.get("name") == field_name:
            return field.get("value"), False
    return None, False


def corrected_copy(record: Dict[str, Any]) -> Dict[str, Any]:
    copied = copy.deepcopy(record)
    for field in copied.get("fields") or []:
        correction = latest_correction(copied, field["name"])
        if correction:
            field["value"] = correction.get("new")
            field["display"] = correction.get("new")
            field["status"] = "confirmed"
    return copied


def apply_corrections(record: Dict[str, Any], spec: Optional[FormSpec] = None,
                      mask_sensitive: bool = False) -> Dict[str, Any]:
    """Alias for producing a corrected copy; records are never mutated."""
    copied = corrected_copy(record)
    if spec is not None:
        for field in copied.get("fields") or []:
            fs = spec.field_by_name(field["name"])
            if fs is not None:
                field["sensitive"] = fs.sensitive
                field["label"] = fs.label_for(copied.get("language") or "en")
    return copied


def field_columns(spec: Optional[FormSpec]) -> List[Tuple[str, str, bool]]:
    if spec is None:
        return []
    language = spec.language
    return [(f.name, f.label_for(language), f.sensitive) for f in spec.fields]


def response_rows(records: Iterable[Dict[str, Any]], spec: Optional[FormSpec] = None,
                  mask_sensitive: bool = False) -> List[Dict[str, Any]]:
    corrected = [corrected_copy(r) for r in records]
    rows = []
    for record, copied in zip(records, corrected):
        progress = record.get("progress") or [0, 0]
        row: Dict[str, Any] = {
            "call_id": record.get("call_id"),
            "time": record.get("started_at"),
            "language": record.get("language"),
            "outcome": record.get("outcome"),
            "answered": f"{progress[0]}/{progress[1]}",
        }
        fields = copied.get("fields") or []
        by_name = {f.get("name"): f for f in fields}
        names = [f.name for f in spec.fields] if spec is not None else list(by_name)
        for name in names:
            field = by_name.get(name)
            if field is None:
                row[name] = ""
                continue
            status = field.get("status")
            if status == "confirmed":
                value = field.get("value")
                value = mask_sensitive_value(name, value, bool(field.get("sensitive")) and mask_sensitive)
                if latest_correction(record, name):
                    value = f"{value} ✎" if value not in (None, "") else "✎"
            elif status in ("skipped", "needs_help"):
                value = f"({status})"
            else:
                value = ""
            row[name] = value
        rows.append(row)
    return rows


def to_rows(records: Iterable[Dict[str, Any]], spec: Optional[FormSpec] = None,
            mask_sensitive: bool = False) -> List[Dict[str, Any]]:
    return response_rows(records, spec, mask_sensitive)


def paginate(records: Iterable[Dict[str, Any]], page: int = 1,
             page_size: int = 25) -> Dict[str, Any]:
    items = list(records)
    page_size = max(1, int(page_size))
    pages = max(1, (len(items) + page_size - 1) // page_size)
    page = min(max(1, int(page)), pages)
    start = (page - 1) * page_size
    return {"rows": items[start:start + page_size], "page": page, "pages": pages,
            "total": len(items), "start": start}


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    if not rows:
        return "\ufeff" + _export_csv([])
    cols = list(rows[0])
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + buf.getvalue()


def to_csv(rows: List[Dict[str, Any]]) -> str:
    return rows_to_csv(rows)


def export_csv(records: Iterable[Dict[str, Any]], spec: Optional[FormSpec] = None,
               mask_sensitive: bool = False) -> str:
    return rows_to_csv(response_rows(records, spec, mask_sensitive))


def csv_bytes(records: Iterable[Dict[str, Any]], spec: Optional[FormSpec] = None,
              mask_sensitive: bool = False) -> bytes:
    return export_csv(records, spec, mask_sensitive).encode("utf-8")


def export_json(record: Dict[str, Any]) -> str:
    import json
    return json.dumps(record, ensure_ascii=False, indent=2)


def add_correction(store: CallStore, call_id: str, field_name: str, new_text: str,
                   by: str = "admin", ex: Any = None) -> Dict[str, Any]:
    record = store.load(call_id)
    if record.get("outcome") == "in_progress":
        raise ValueError("An in-progress call cannot be modified.")
    extractor_obj = ex or extractor
    if hasattr(extractor_obj, "extract_field_full"):
        result = extractor_obj.extract_field_full(new_text, field_name)
    elif callable(extractor_obj):
        result = extractor_obj(new_text, field_name)
    else:
        raise ValueError("A field extractor is not available.")
    status = getattr(result, "status", None)
    if status not in ("ok", "needs_check"):
        issues = getattr(result, "issues", None) or []
        raise ValueError("; ".join(str(issue) for issue in issues) or "The correction could not be validated.")
    value = getattr(result, "value", None)
    if value is None:
        raise ValueError("The correction did not produce a value.")

    old = None
    for field in record.get("fields") or []:
        if field.get("name") == field_name:
            previous = latest_correction(record, field_name)
            old = previous["new"] if previous else field.get("value")
            break
    at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    corrections = record.setdefault("corrections", [])
    corrections.append({
        "field": field_name,
        "old": old,
        "new": value,
        "at": at,
        "by": by,
    })
    record["reviewed"] = True
    store.save(record)
    return record


def add_admin_correction(store: CallStore, call_id: str, field_name: str, new_text: str,
                         ex: Any = None) -> Dict[str, Any]:
    return add_correction(store, call_id, field_name, new_text, ex=ex)


def delete_call(store: CallStore, call_id: str) -> bool:
    return store.delete(call_id)


def delete_form_calls(store: CallStore, form_id: str) -> int:
    deleted = 0
    for summary in store.list():
        if summary.get("form_id") == form_id:
            call_id = summary.get("call_id")
            if call_id and store.delete(call_id):
                deleted += 1
    return deleted


def audio_turns(store: CallStore, call_id: str) -> List[str]:
    folder = os.path.join(store.dir, call_id)
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.startswith("turn_") and name.lower().endswith(".wav")
    ]
