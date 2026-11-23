"""
export.py -- turn stored calls into a table (for the Admin page, a CSV download, or the Analytics tab)
======================================================================================================

    rows = to_rows(calls.list_records("clinic_registration"))     # one dict per call: call_id, outcome, one column per field
    csv_text = to_csv(rows)
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, Iterable, List, Optional

from callpage.store import CallStore


def records(store: CallStore, form_id: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for s in store.list():
        if form_id and s.get("form_id") != form_id:
            continue
        try:
            out.append(store.load(s["call_id"]))
        except Exception:                                   # noqa: BLE001 - one unreadable file must not hide the others
            continue
    return out


def to_rows(recs: Iterable[Dict[str, Any]], mask_sensitive: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for r in recs:
        row: Dict[str, Any] = {"call_id": r["call_id"], "form": r.get("form_title"), "language": r.get("language"),
                               "started_at": r.get("started_at"), "outcome": r.get("outcome"),
                               "consent": r.get("consent"), "answered": f"{r['progress'][0]}/{r['progress'][1]}"}
        for f in r.get("fields", []):
            v = f.get("value")
            if mask_sensitive and f.get("sensitive") and v:
                v = "*" * max(0, len(v) - 2) + v[-2:]
            row[f["name"]] = v if f.get("status") == "confirmed" else ("" if f.get("status") == "pending" else f"({f.get('status')})")
        rows.append(row)
    return rows


def to_csv(rows: List[Dict[str, Any]]) -> str:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    return buf.getvalue()
