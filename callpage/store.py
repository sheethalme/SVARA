"""
store.py -- saves every call (answers, confirmations, timestamps) as one JSON file: data/calls/<call_id>.json
=============================================================================================================

* The file is rewritten after EVERY turn (atomically: write to a temp file, then rename), so a call that drops
  half way still has its answers on disk and a crash can never leave a half-written file.
* ``call_id`` is validated (hex only), so a URL like ``/api/call/../../secret`` cannot reach other files.
* Audio is only written when the form allows it AND the caller agreed (see ``CallSession.consent``).
* PRIVACY: these files contain personal (health) data in plain text.  Keep the folder out of git (it is in
  .gitignore), restrict who can read it, and delete old calls (``delete`` / ``purge_older_than``).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List

_ID = re.compile(r"^[a-f0-9]{8,32}$")


class StoreError(ValueError):
    pass


class CallStore:
    def __init__(self, directory: str):
        self.dir = directory

    def _check(self, call_id: str) -> str:
        if not isinstance(call_id, str) or not _ID.match(call_id):
            raise StoreError("invalid call id")
        return call_id

    def _path(self, call_id: str) -> str:
        return os.path.join(self.dir, self._check(call_id) + ".json")

    def save(self, record: Dict[str, Any]) -> None:
        path = self._path(record["call_id"])
        os.makedirs(self.dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def load(self, call_id: str) -> Dict[str, Any]:
        try:
            with open(self._path(call_id), encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            raise StoreError("call not found") from None

    def list(self) -> List[Dict[str, Any]]:
        """Short summaries of all calls, newest first."""
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(self.dir):
            return out
        for fn in os.listdir(self.dir):
            if not fn.endswith(".json"):
                continue
            try:
                r = self.load(fn[:-5])
            except (StoreError, OSError, ValueError):
                continue
            out.append({k: r.get(k) for k in ("call_id", "form_id", "form_title", "language", "started_at", "ended_at",
                                                "outcome", "progress")})
        return sorted(out, key=lambda r: r.get("started_at") or "", reverse=True)

    def save_audio(self, call_id: str, n: int, wav: bytes) -> str:
        folder = os.path.join(self.dir, self._check(call_id))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"turn_{int(n):03d}.wav")
        with open(path, "wb") as fh:
            fh.write(wav)
        return path

    def delete(self, call_id: str) -> bool:
        removed = False
        p = self._path(call_id)
        if os.path.exists(p):
            os.remove(p); removed = True
        folder = os.path.join(self.dir, call_id)
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                os.remove(os.path.join(folder, fn))
            os.rmdir(folder); removed = True
        return removed

    def purge_older_than(self, days: float) -> int:
        """Delete calls whose file is older than ``days`` days; returns how many."""
        cutoff, n = time.time() - days * 86400, 0
        for r in self.list():
            p = self._path(r["call_id"])
            if os.path.exists(p) and os.path.getmtime(p) < cutoff:
                n += self.delete(r["call_id"])
        return n
