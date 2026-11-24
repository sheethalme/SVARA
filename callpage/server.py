"""
server.py -- the call page's web server (standard library only; no extra packages)
==================================================================================

    python -m callpage                       # then open http://127.0.0.1:8765
    python -m callpage --forms data/forms --calls data/calls --port 8765

It serves ONE web page (``static/index.html``) and a small JSON API:

    GET  /                       the call page (a per-run security token is written into it)
    GET  /healthz                "ok"
    GET  /api/forms              the forms the admin created (id, title, languages, fields)
    POST /api/call/start         {"form_id": "...", "language": "en"}   -> first turn
    POST /api/call/<id>/turn     body = WAV audio of ONE utterance        -> next turn
    POST /api/call/<id>/event    {"event": "no_speech" | "hangup"}        -> next turn
    GET  /api/call/<id>          the stored record of the call (answers, confirmations ...)
    GET  /api/calls              summaries of all calls
    POST /api/tts                {"text": "...", "lang": "hi"}            -> MP3 (503: use the browser voice)

Safety (it handles personal health data and listens on your laptop only):
  * binds to 127.0.0.1 by default; the Host header must be localhost / 127.0.0.1 (stops DNS-rebinding);
  * every /api request needs the random token that only the served page knows (stops other websites from
    posting to your local server);
  * request sizes are limited; call ids are hex only; errors never include stack traces.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import threading
import time
import extractor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

from callpage import prompts, tts as tts_mod
from callpage.auth import Auth, AuthError
from callpage.forms import FormError, FormSpec, FormStore, build_form_dict, slugify
from callpage.session import CallSession, Transcriber
from callpage.store import CallStore, StoreError

log = logging.getLogger("callpage")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_JSON = 64 * 1024
MAX_AUDIO = 6 * 1024 * 1024
MAX_PDF = 12 * 1024 * 1024
_CALL_ID = re.compile(r"^[a-f0-9]{8,32}$")


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra: Any):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


# ============================================================================== the application (no HTTP in here)
class CallApp:
    def __init__(self, forms: FormStore, calls: CallStore, transcriber: Transcriber,
                 tts_fn: Optional[Callable[[str, str], bytes]] = None, session_ttl_s: float = 1800.0,
                 clock: Callable[[], float] = time.time, templates: Optional[Any] = None,
                 auth: Optional[Auth] = None):
        self.forms, self.calls, self.transcriber, self.tts_fn = forms, calls, transcriber, tts_fn
        self._templates = templates
        self.auth = auth                      # None = no accounts (tests, simple local use)
        self.ttl, self.clock = session_ttl_s, clock
        self._sessions: Dict[str, Tuple[CallSession, threading.Lock, list]] = {}
        self._lock = threading.Lock()

    # ---- accounts, roles and ownership
    def need(self, user: Optional[Dict[str, Any]], *roles: str) -> None:
        if self.auth is None:
            return
        if user is None or user["role"] not in roles:
            raise ApiError(403, "That action is only for administrators." if roles == ("admin",) else "Please sign in.")

    def _owner(self, kind: str, ident: str) -> Optional[str]:
        return self.auth.owners.get(kind, ident) if self.auth else None

    def _can_see_form(self, form_id: str, user: Optional[Dict[str, Any]]) -> bool:
        owner = self._owner("form", form_id)
        return owner is None or (user is not None and owner == user["username"])

    def _locked(self, form_id: str, user: Optional[Dict[str, Any]]) -> bool:
        """True when the form has an access code this caller has not entered yet (admins and the form's owner are exempt)."""
        if self.auth is None or (user and user["role"] == "admin"):
            return False
        if user and self._owner("form", form_id) == user["username"]:
            return False
        return not (user and self.auth.access.is_unlocked(user["username"], form_id))

    def _own_call(self, call_id: str, user: Optional[Dict[str, Any]]) -> None:
        if self.auth is None or (user and user["role"] == "admin"):
            return
        if user is None or self._owner("call", call_id) != user["username"]:
            raise ApiError(404, "call not found or already ended")

    def account(self, action: str, d: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
        if self.auth is None:
            raise ApiError(404, "accounts are not enabled on this server")
        users, sessions = self.auth.users, self.auth.sessions
        try:
            if action == "status":
                return {"admin_exists": users.admin_exists()}
            if action == "signup":                    # creating an account does not sign you in: sign in next
                return {"user": users.signup(str(d.get("name", "")), str(d.get("username", "")), d.get("password"),
                                             "admin" if d.get("role") == "admin" else "user")}
            if action == "login":
                role = d.get("role") if d.get("role") in ("admin", "user") else None
                user = users.login(str(d.get("username") or ""), d.get("password"), role)
                return {"user": user, "token": sessions.create(user)}
            if action == "me":
                user = sessions.get(token)
                if user is None:
                    raise ApiError(401, "Your session has ended. Please sign in again.", session_expired=True)
                return {"user": user}
            if action == "logout":
                sessions.drop(token)
                return {"ok": True}
        except AuthError as exc:
            raise ApiError(exc.status, exc.message, **exc.extra) from None
        raise ApiError(404, "not found")

    # ---- admin: upload a PDF form, read its fields, publish it as a voice form
    @property
    def templates(self):
        if self._templates is None:
            from formreader.store import TemplateStore
            self._templates = TemplateStore(os.path.join(os.path.dirname(self.forms.dir.rstrip("/\\")) or ".", "templates"))
        return self._templates

    def _unique_id(self, base: str) -> str:
        taken = {fid for fid, _ in self.forms.list()} | {tid for tid, _ in self.templates.list()}
        if base not in taken:
            return base
        n = 2
        while f"{base}_{n}" in taken:
            n += 1
        return f"{base}_{n}"

    def analyze_form(self, pdf: bytes, filename: str, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from formreader import semantics
        from formreader.analyze import analyze_pdf
        from formreader.model import TemplateError
        if not pdf.startswith(b"%PDF"):
            raise ApiError(400, "That file is not a PDF. Please choose a PDF form.")
        try:
            t = analyze_pdf(pdf, filename=filename)
            t.id = self._unique_id(t.id)
            self.templates.save(t, pdf)
            if self.auth and user:
                self.auth.owners.set("tpl", t.id, user["username"])
        except TemplateError as exc:
            raise ApiError(422, str(exc)) from None
        except Exception as exc:                                   # noqa: BLE001 - damaged / encrypted / scanned PDF
            log.warning("analyze failed: %s", type(exc).__name__)
            raise ApiError(422, "SVARA could not read that PDF. It must be a digital (not scanned) form.") from None
        return {"template_id": t.id, "title": t.title, "source_name": t.source_name, "acroform": t.acroform,
                "pages": len(t.pages), "warnings": t.warnings, "kinds": sorted(extractor.RULES),
                "fields": [{"id": f.id, "label": f.label, "page": f.page + 1, "shape": f.shape, "call_kind": f.call_kind,
                            "call_name": f.call_name, "required": f.required, "ignored": f.ignored,
                            "question": semantics.make_question(f) if not f.ignored else "",
                            "confidence": round(f.confidence, 2), "options": [o.label for o in f.options], "notes": f.notes}
                           for f in t.fields]}

    def publish_form(self, data: Dict[str, Any], user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from formreader.model import TemplateError
        from formreader.publish import publish_template
        tid = str(data.get("template_id", ""))
        if self.auth and (user is None or self._owner("tpl", tid) != user["username"]):
            raise ApiError(403, "That uploaded form belongs to someone else.")
        try:
            t = self.templates.load(tid)
            edits = {str(e.get("id")): e for e in data.get("fields", []) if isinstance(e, dict)}
            if isinstance(data.get("title"), str) and data["title"].strip():
                t.title = data["title"].strip()[:120]
            for f in t.fields:
                e = edits.get(f.id)
                if not e:
                    continue
                if isinstance(e.get("label"), str) and e["label"].strip():
                    f.label = e["label"].strip()[:120]
                if isinstance(e.get("call_name"), str) and e["call_name"].strip():
                    f.call_name = re.sub(r"[\s\-]+", "_", e["call_name"].strip().lower())
                if e.get("call_kind") in extractor.RULES:
                    f.call_kind = e["call_kind"]
                if isinstance(e.get("required"), bool):
                    f.required = e["required"]
                if isinstance(e.get("ignored"), bool):
                    f.ignored = e["ignored"]
                if isinstance(e.get("question"), str):
                    f.question = e["question"].strip()[:300] or None
            spec = publish_template(t, self.forms)
            self.templates.save(t)
            if self.auth and user and user["role"] == "user":
                self.auth.owners.set("form", spec.id, user["username"])          # a caller's own form is private to them
        except (TemplateError, FormError) as exc:
            raise ApiError(400, str(exc)) from None
        out = self._form_json(spec)
        if self.auth and user and user["role"] == "admin":
            out["access_code"] = self._ensure_code(spec.id)
        return out

    def save_form(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create / update a form written by hand in the form studio."""
        title = str(data.get("title", "")).strip()
        if not title:
            raise ApiError(400, "Give the form a name.")
        kinds = {"yesno": "yes_no"}
        fields: list = []
        used: set = set()
        for i, f in enumerate(data.get("fields") or []):
            if not isinstance(f, dict):
                continue
            label = str(f.get("label") or "").strip()
            name = str(f.get("name") or "").strip()
            if name.startswith("custom_") and label:
                name = slugify(label)[:40]
            base, n = name, 2
            while name in used:
                name, n = f"{base}_{n}", n + 1
            used.add(name)
            item: Dict[str, Any] = {"name": name, "required": bool(f.get("required", True)),
                                    "kind": kinds.get(f.get("kind"), f.get("kind") or "text")}
            if label:
                item["label"] = {"en": label}
            if str(f.get("question") or "").strip():
                item["question"] = {"en": str(f["question"]).strip()}
            fields.append(item)
        if not fields:
            raise ApiError(400, "Add at least one question.")
        langs = [l for l in (data.get("languages") or ["en"]) if l in prompts.LANGS] or ["en"]
        wanted = str(data.get("id") or "")
        known = {fid for fid, _ in self.forms.list()}
        form_id = wanted if wanted in known else self._unique_id(slugify(title))
        try:
            settings = self.forms.get(form_id).settings if form_id in known else None
            spec = self.forms.save(build_form_dict(title, fields, "en", langs, settings, form_id))
        except FormError as exc:
            raise ApiError(400, str(exc)) from None
        out = self._form_json(spec)
        if self.auth:
            out["access_code"] = self._ensure_code(spec.id) if form_id not in known else self.auth.access.code_of(spec.id)
        return out

    @staticmethod
    def _form_json(f: FormSpec) -> Dict[str, Any]:
        return {"id": f.id, "title": f.title, "language": f.language, "languages": f.languages, "settings": f.settings,
                "fields": [{"name": x.name, "label": x.label_for(f.language), "required": x.required, "kind": x.kind,
                            **({"question": x.question.get("en", "")} if x.question else {})} for x in f.fields]}

    def filled_pdf(self, call_id: str, user: Optional[Dict[str, Any]] = None) -> bytes:
        self._own_call(call_id, user)
        from formreader.model import TemplateError
        from formreader.publish import fill_from_record
        rec = self.record(call_id)
        try:
            fid = rec.get("form_id", "")
            return fill_from_record(self.templates.load(fid), self.templates.pdf_bytes(fid), rec)[0]
        except (TemplateError, OSError):
            raise ApiError(404, "This form was not made from an uploaded PDF, so there is nothing to fill in.") from None

    # ---- access codes
    def unlock_by_code(self, code: str, user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if self.auth is None or user is None:
            raise ApiError(403, "Please sign in.")
        try:
            ids = self.auth.access.unlock_by_code(user["username"], code)
        except AuthError as exc:
            raise ApiError(exc.status, exc.message, **exc.extra) from None
        titles = dict(self.forms.list())
        found = [{"id": fid, "title": titles[fid]} for fid in ids if fid in titles and self._can_see_form(fid, user)]
        if not found:                                       # the code belonged to a form that no longer exists
            raise ApiError(403, "That access code is not right.", code_wrong=True)
        return {"forms": found}

    def set_access(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fid = str(data.get("id", ""))
        if self.auth is None:
            raise ApiError(404, "accounts are not enabled on this server")
        if fid not in {x for x, _ in self.forms.list()} or self._owner("form", fid) is not None:
            raise ApiError(404, "That form no longer exists.")
        action = data.get("action")
        try:
            if action == "generate":
                code = self.auth.access.set_code(fid, self.auth.access.generate())
            elif action == "set":
                code = self.auth.access.set_code(fid, str(data.get("code", "")))
            elif action == "clear":
                code = self.auth.access.set_code(fid, None)
            else:
                raise ApiError(400, "unknown action")
        except AuthError as exc:
            raise ApiError(exc.status, exc.message) from None
        return {"id": fid, "access_code": code}

    def _ensure_code(self, form_id: str) -> Optional[str]:
        """Every form the administrator publishes gets an access code automatically (they can change or remove it)."""
        if self.auth is None:
            return None
        return self.auth.access.code_of(form_id) or self.auth.access.set_code(form_id, self.auth.access.generate())

    # ---- admin: delete a form, per-form privacy settings, server status
    def delete_form(self, form_id: str, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        known = {fid for fid, _ in self.forms.list()}
        if form_id not in known:
            raise ApiError(404, "That form no longer exists.")
        owner = self._owner("form", form_id)
        if self.auth:
            if user is None:
                raise ApiError(403, "Please sign in.")
            if user["role"] == "user" and owner != user["username"]:
                raise ApiError(403, "You can only delete forms you uploaded yourself.")
            if user["role"] == "admin" and owner is not None:
                raise ApiError(404, "That form no longer exists.")
        shared = {fid for fid in known if self._owner("form", fid) is None}
        if owner is None and len(shared) <= 1:
            raise ApiError(400, "Keep at least one form: callers need something to fill in.")
        try:
            os.remove(self.forms._path(form_id))
        except (OSError, FormError):
            raise ApiError(500, "The form file could not be deleted.") from None
        try:                                   # the uploaded PDF and its template (only exist for PDF forms)
            self.templates.delete(form_id)
        except Exception:                      # noqa: BLE001
            pass
        if self.auth:
            self.auth.owners.drop("form", form_id)
            self.auth.owners.drop("tpl", form_id)
            self.auth.access.forget(form_id)
        return {"deleted": form_id}

    def form_settings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            spec = self.forms.get(str(data.get("id", "")))
        except FormError:
            raise ApiError(404, "That form no longer exists.") from None
        incoming = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        d = spec.to_dict()
        new = dict(d["settings"])
        for key in ("require_consent", "store_audio", "redact_sensitive_log"):
            if isinstance(incoming.get(key), bool):
                new[key] = incoming[key]
        attempts = incoming.get("max_attempts")
        if isinstance(attempts, int) and not isinstance(attempts, bool) and 1 <= attempts <= 6:
            new["max_attempts"] = attempts
        d["settings"] = new
        try:
            saved = self.forms.save(d)
        except FormError as exc:
            raise ApiError(400, str(exc)) from None
        return {"id": saved.id, "settings": saved.settings}

    def server_status(self) -> Dict[str, Any]:
        speech: Dict[str, Any] = {}
        try:
            import stt
            engine = stt.get_default()
            speech = {"cloud_ready": bool(engine.cloud.available), "provider": engine.cfg.cloud_provider,
                      "local_model": engine.cfg.local_model, "beam": engine.cfg.beam_size, "threads": engine.cfg.cpu_threads}
        except Exception:                                          # noqa: BLE001 - status is informational
            speech = {"error": "speech engine not loaded"}
        return {"speech": speech, "server_voice": self.tts_fn is not None, "forms": len(self.forms.list()),
                "calls": len(self.calls.list()), "forms_dir": os.path.abspath(self.forms.dir), "calls_dir": os.path.abspath(self.calls.dir)}

    # ---- forms
    def list_forms(self, user: Optional[Dict[str, Any]] = None):
        out = []
        for fid, _ in self.forms.list():
            if not self._can_see_form(fid, user) or self._locked(fid, user):
                continue                                                # no code entered -> the caller does not even see it
            item = self._form_json(self.forms.get(fid))
            item["mine"] = bool(self.auth and user and self._owner("form", fid) == user["username"])
            item["has_pdf"] = os.path.isfile(os.path.join(self.templates.dir, fid, "source.pdf"))
            if self.auth:
                code = self.auth.access.code_of(fid)
                item["has_code"] = code is not None
                if user and user["role"] == "admin":
                    item["access_code"] = code                          # only the administrator ever sees a code
            out.append(item)
        return out

    # ---- calls
    def _gc(self) -> None:
        now = self.clock()
        with self._lock:
            for cid in [c for c, (_, _, t) in self._sessions.items() if now - t[0] > self.ttl]:
                del self._sessions[cid]

    def start_call(self, form_id: str, language: Optional[str] = None, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._can_see_form(form_id, user):
            raise ApiError(404, "form not found")
        if self._locked(form_id, user):
            raise ApiError(404, "form not found")                       # same answer as for a form that does not exist
        try:
            form = self.forms.get(form_id)
        except FormError as exc:
            raise ApiError(404, str(exc)) from None
        lang = prompts.norm_lang(language) if language else form.language
        if lang not in form.languages:
            raise ApiError(400, f"language '{lang}' is not offered by this form")
        self._gc()
        s = CallSession(form, lang, transcriber=self.transcriber, store=self.calls, clock=self.clock)
        turn = s.start()
        if self.auth and user:
            self.auth.owners.set("call", s.call_id, user["username"])
        with self._lock:
            self._sessions[s.call_id] = (s, threading.Lock(), [self.clock()])
        return {"call_id": s.call_id, "turn": turn.to_dict(), "form": {"id": form.id, "title": form.title}}

    def _get(self, call_id: str):
        if not _CALL_ID.match(call_id or ""):
            raise ApiError(404, "call not found")
        with self._lock:
            item = self._sessions.get(call_id)
        if item is None:
            raise ApiError(404, "call not found or already ended")
        item[2][0] = self.clock()
        return item

    def turn(self, call_id: str, wav: bytes, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._own_call(call_id, user)
        s, lock, _ = self._get(call_id)
        if len(wav) < 44:
            raise ApiError(400, "that is not a WAV file")
        with lock:
            if s.settings.get("store_audio") and s.consent:
                try:
                    self.calls.save_audio(s.call_id, s.audio_turns + 1, wav)
                except OSError:
                    pass
            return self._finish(s, s.handle_audio(wav))

    def event(self, call_id: str, name: str, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._own_call(call_id, user)
        s, lock, _ = self._get(call_id)
        with lock:
            if name == "no_speech":
                return self._finish(s, s.handle_no_speech())
            if name == "hangup":
                return self._finish(s, s.hangup())
        raise ApiError(400, "unknown event")

    def _finish(self, s: CallSession, turn) -> Dict[str, Any]:
        if turn.done:
            with self._lock:
                self._sessions.pop(s.call_id, None)
        return {"call_id": s.call_id, "turn": turn.to_dict()}

    def record(self, call_id: str) -> Dict[str, Any]:
        try:
            return self.calls.load(call_id)
        except StoreError as exc:
            raise ApiError(404, str(exc)) from None

    def synth(self, text: str, lang: str) -> bytes:
        if self.tts_fn is None:
            raise ApiError(503, "no server voice configured")
        try:
            return self.tts_fn(text, prompts.norm_lang(lang))
        except tts_mod.TTSUnavailable as exc:
            raise ApiError(503, str(exc)) from None


# ============================================================================== HTTP layer
def make_handler(app: CallApp, token: str, allowed_hosts: Tuple[str, ...]):
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
        page = fh.read()

    class Handler(BaseHTTPRequestHandler):
        server_version = "SVARACall/1.0"

        def log_message(self, fmt, *args):             # keep personal data out of logs: no bodies, no paths with ids
            log.debug("%s %s", self.command, urlparse(self.path).path.split("/")[1:3])

        # ---- helpers
        def _send(self, status: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self, limit: int) -> bytes:
            n = self.headers.get("Content-Length")
            if n is None or not n.isdigit():
                raise ApiError(411, "Content-Length is required")
            if int(n) > limit:
                raise ApiError(413, "request too large")
            return self.rfile.read(int(n))

        def _check_host(self) -> None:
            host = (self.headers.get("Host") or "").lower()
            if host not in allowed_hosts:
                raise ApiError(403, "forbidden host")

        def _check_token(self) -> None:
            sent = self.headers.get("X-Call-Token") or ""
            if not secrets.compare_digest(sent.encode(), token.encode()):
                raise ApiError(401, "missing or wrong token", stale_page=True)

        def _route(self, method: str) -> None:
            try:
                self._check_host()
                path = urlparse(self.path).path
                if method == "GET" and path in ("/", "/index.html", "/admin", "/caller", "/forms", "/responses", "/settings",
                                                 "/login", "/login/admin", "/login/caller"):
                    html = page.replace("__TOKEN__", token)
                    return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                                      {"Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline' blob:; worker-src blob:; "
                                       "style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:"})
                if method == "GET" and path == "/healthz":
                    return self._send(200, b"ok", "text/plain")
                if not path.startswith("/api/"):
                    raise ApiError(404, "not found")
                self._check_token()
                parts = path.strip("/").split("/")[1:]                     # after "api"
                session_token = self.headers.get("X-Session") or None
                if parts[:1] == ["auth"] and len(parts) == 2:
                    if method == "POST":
                        return self._json(200, app.account(parts[1], self._json_body(), session_token))
                    if method == "GET" and parts[1] in ("me", "status"):
                        return self._json(200, app.account(parts[1], {}, session_token))
                    raise ApiError(404, "not found")
                user = None
                if app.auth is not None:
                    user = app.auth.sessions.get(session_token)
                    if user is None:
                        raise ApiError(401, "Your session has ended. Please sign in again.", session_expired=True)
                admin = lambda: app.need(user, "admin")                    # noqa: E731
                if method == "GET" and parts == ["forms"]:
                    return self._json(200, app.list_forms(user))
                if method == "GET" and parts == ["calls"]:
                    admin()
                    return self._json(200, app.calls.list())
                if method == "GET" and len(parts) == 2 and parts[0] == "call":
                    admin()
                    return self._json(200, app.record(parts[1]))
                if method == "POST" and parts == ["call", "start"]:
                    d = self._json_body()
                    return self._json(200, app.start_call(str(d.get("form_id", "")), d.get("language"), user))
                if method == "POST" and len(parts) == 3 and parts[0] == "call" and parts[2] == "turn":
                    return self._json(200, app.turn(parts[1], self._body(MAX_AUDIO), user))
                if method == "POST" and len(parts) == 3 and parts[0] == "call" and parts[2] == "event":
                    return self._json(200, app.event(parts[1], str(self._json_body().get("event", "")), user))
                if method == "POST" and parts == ["forms", "unlock-by-code"]:
                    return self._json(200, app.unlock_by_code(str(self._json_body().get("code", "")), user))
                if method == "POST" and parts == ["admin", "forms", "access"]:
                    admin()
                    return self._json(200, app.set_access(self._json_body()))
                if method == "POST" and parts == ["admin", "forms", "analyze"]:
                    name = re.sub(r"[^\w .()-]", "_", self.headers.get("X-Filename") or "form.pdf")[:120]
                    return self._json(200, app.analyze_form(self._body(MAX_PDF), name, user))
                if method == "POST" and parts == ["admin", "forms", "publish"]:
                    return self._json(200, app.publish_form(self._json_body(MAX_PDF), user))
                if method == "POST" and parts == ["admin", "forms", "delete"]:
                    return self._json(200, app.delete_form(str(self._json_body().get("id", "")), user))
                if method == "POST" and parts == ["admin", "forms", "save"]:
                    admin()
                    return self._json(200, app.save_form(self._json_body()))
                if method == "POST" and parts == ["admin", "forms", "settings"]:
                    admin()
                    return self._json(200, app.form_settings(self._json_body()))
                if method == "GET" and parts == ["admin", "settings"]:
                    admin()
                    return self._json(200, app.server_status())
                if method == "GET" and len(parts) == 3 and parts[0] == "call" and parts[2] == "filled.pdf":
                    return self._send(200, app.filled_pdf(parts[1], user), "application/pdf",
                                      {"Content-Disposition": "attachment; filename=filled_form.pdf"})
                if method == "POST" and parts == ["tts"]:
                    d = self._json_body()
                    data = app.synth(str(d.get("text", "")), str(d.get("lang", "en")))
                    return self._send(200, data, "audio/mpeg")
                raise ApiError(404, "not found")
            except ApiError as exc:
                self._json(exc.status, {"error": exc.message, **exc.extra})
            except ConnectionError:                       # the browser went away (Windows: ConnectionAbortedError): ignore
                pass
            except Exception:                                              # noqa: BLE001 - never leak a traceback
                log.exception("unhandled error")
                try:
                    self._json(500, {"error": "internal error"})
                except OSError:
                    pass

        def _json_body(self, limit: int = MAX_JSON) -> Dict[str, Any]:
            raw = self._body(limit)
            try:
                d = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                raise ApiError(400, "invalid JSON") from None
            if not isinstance(d, dict):
                raise ApiError(400, "JSON object expected")
            return d

        def do_GET(self): self._route("GET")
        def do_POST(self): self._route("POST")

    return Handler


class CallServer:
    """A running server (used by ``python -m callpage`` and by the tests)."""

    def __init__(self, app: CallApp, host: str = "127.0.0.1", port: int = 8765, token: Optional[str] = None):
        self.token = token or secrets.token_urlsafe(24)
        self.app = app
        probe = ThreadingHTTPServer((host, port), BaseHTTPRequestHandler)      # find the real port when port == 0
        self.port = probe.server_address[1]
        probe.server_close()
        hosts = tuple({f"localhost:{self.port}", f"127.0.0.1:{self.port}", f"{host}:{self.port}"})
        self.httpd = ThreadingHTTPServer((host, self.port), make_handler(app, self.token, hosts))
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.port}/"
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CallServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown(); self.httpd.server_close()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()


def build_app(forms_dir: str, calls_dir: str, preprocess: Optional[str] = None, engine: Optional[str] = None,
              tts_cache: Optional[str] = None, use_tts: bool = True, transcriber: Optional[Transcriber] = None,
              accounts: bool = False) -> CallApp:
    from callpage.pipeline import make_transcriber
    tts_fn = None
    if use_tts:
        tts_fn = lambda text, lang: tts_mod.synth(text, lang, tts_cache or os.path.join(os.path.dirname(calls_dir.rstrip("/\\")) or ".", "tts_cache"))
    auth = Auth(os.path.dirname(os.path.abspath(forms_dir.rstrip("/\\"))) or ".") if accounts else None
    return CallApp(FormStore(forms_dir), CallStore(calls_dir), transcriber or make_transcriber(preprocess, engine), tts_fn, auth=auth)


def warm_up() -> None:
    """Load the speech model and run one tiny transcription in the background, so the FIRST question of the first
    call is not slowed down by the model loading (the biggest single delay on a laptop)."""
    def work() -> None:
        try:
            import numpy as np
            import stt
            engine = stt.get_default()
            if getattr(engine.local, "available", True) and not engine.cloud.available:
                engine.local.preload(include_specialists=False)
                engine.transcribe_full(np.zeros(16000, dtype="float32") + 1e-4, 16000, language="en", engine="local")
            log.info("speech model is warm")
        except Exception as exc:                                     # noqa: BLE001 - warming up is only an optimisation
            log.info("warm-up skipped: %s", type(exc).__name__)
    threading.Thread(target=work, daemon=True, name="svara-warmup").start()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SVARA hands-free voice call page")
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless you know what you are doing")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--forms", default=os.path.join("data", "forms"))
    ap.add_argument("--calls", default=os.path.join("data", "calls"))
    ap.add_argument("--preprocess", choices=["off", "light", "full"], default=None)
    ap.add_argument("--engine", choices=["auto", "cloud", "local"], default=None)
    ap.add_argument("--no-tts", action="store_true", help="never use the server voice (gTTS); use the browser voice")
    ap.add_argument("--script", help="TEST ONLY: JSON list of transcripts to return instead of real speech recognition")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--fast", action="store_true", help="use the small-but-quick local model (base): ~3x faster, less accurate (esp. Indian languages)")
    ap.add_argument("--no-accounts", action="store_true", help="TEST/DEMO ONLY: no sign-in, everyone is allowed everything")
    ap.add_argument("--quality", action="store_true", help="slower but slightly more accurate speech recognition (beam search)")
    a = ap.parse_args(argv)
    os.environ.setdefault("SVARA_STT_BEAM", "5" if a.quality else "1")                 # greedy decoding: ~3-5x faster
    os.environ.setdefault("SVARA_STT_THREADS", str(max(4, (os.cpu_count() or 8) // 2)))   # measured: half the logical cores is fastest
    if a.fast:
        os.environ.setdefault("SVARA_WHISPER_MODEL", "base")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:                                                   # read GROQ_API_KEY etc. from .env (as in stt.py)
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:                                      # noqa: BLE001
        pass
    transcriber = None
    if a.script:
        from callpage.testing import scripted_transcriber
        with open(a.script, encoding="utf-8") as fh:
            transcriber = scripted_transcriber(json.load(fh))
    app = build_app(a.forms, a.calls, a.preprocess, a.engine, use_tts=not a.no_tts, transcriber=transcriber,
                    accounts=not a.no_accounts and not a.script)
    if not app.forms.list():
        print(f"No forms found in {a.forms}. Create one in the Admin page, or copy data/forms/clinic_registration.json there.")
    srv = CallServer(app, a.host, a.port)
    if not a.script:
        warm_up()
    print(f"SVARA call page: {srv.url}   (Ctrl+C to stop)")
    print(f"forms: {os.path.abspath(a.forms)}   calls are saved to: {os.path.abspath(a.calls)}")
    if a.open:
        import webbrowser
        webbrowser.open(srv.url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
