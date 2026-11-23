"""store.py -- saves reviewed templates: data/templates/<id>/template.json + source.pdf (the blank form)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import List, Tuple

from formreader.model import FormTemplate, TemplateError

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class TemplateStore:
    def __init__(self, directory: str = "data/templates"):
        self.dir = directory

    def _folder(self, tid: str) -> str:
        if not _ID.match(tid or ""):
            raise TemplateError("template id must be lower-case letters, digits, '_' or '-'")
        return os.path.join(self.dir, tid)

    def save(self, template: FormTemplate, pdf_bytes: bytes = b"") -> str:
        template.validate()
        folder = self._folder(template.id)
        os.makedirs(folder, exist_ok=True)
        if pdf_bytes:
            self._write(os.path.join(folder, "source.pdf"), pdf_bytes)
        elif not os.path.exists(os.path.join(folder, "source.pdf")):
            raise TemplateError("the blank PDF must be saved together with the template")
        self._write(os.path.join(folder, "template.json"), template.to_json().encode("utf-8"))
        return folder

    @staticmethod
    def _write(path: str, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def load(self, tid: str) -> FormTemplate:
        path = os.path.join(self._folder(tid), "template.json")
        if not os.path.exists(path):
            raise TemplateError(f"no template '{tid}'")
        with open(path, encoding="utf-8") as fh:
            try:
                return FormTemplate.from_dict(json.load(fh))
            except json.JSONDecodeError as exc:
                raise TemplateError(f"template '{tid}' is damaged: {exc}") from None

    def pdf_bytes(self, tid: str) -> bytes:
        with open(os.path.join(self._folder(tid), "source.pdf"), "rb") as fh:
            return fh.read()

    def list(self) -> List[Tuple[str, str]]:
        out = []
        if os.path.isdir(self.dir):
            for tid in sorted(os.listdir(self.dir)):
                try:
                    out.append((tid, self.load(tid).title))
                except (TemplateError, OSError):
                    continue
        return out

    def delete(self, tid: str) -> bool:
        import shutil
        folder = self._folder(tid)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            return True
        return False
