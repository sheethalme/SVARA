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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

from callpage import prompts, tts as tts_mod
from callpage.forms import FormError, FormSpec, FormStore
from callpage.session import CallSession, Transcriber
from callpage.store import CallStore, StoreError

log = logging.getLogger("callpage")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_JSON = 64 * 1024
MAX_AUDIO = 6 * 1024 * 1024
_CALL_ID = re.compile(r"^[a-f0-9]{8,32}$")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ============================================================================== the application (no HTTP in here)
class CallApp:
    def __init__(self, forms: FormStore, calls: CallStore, transcriber: Transcriber,
                 tts_fn: Optional[Callable[[str, str], bytes]] = None, session_ttl_s: float = 1800.0,
                 clock: Callable[[], float] = time.time):
        self.forms, self.calls, self.transcriber, self.tts_fn = forms, calls, transcriber, tts_fn
        self.ttl, self.clock = session_ttl_s, clock
        self._sessions: Dict[str, Tuple[CallSession, threading.Lock, list]] = {}
        self._lock = threading.Lock()

    # ---- forms
    def list_forms(self):
        out = []
        for fid, _ in self.forms.list():
            f = self.forms.get(fid)
            out.append({"id": f.id, "title": f.title, "language": f.language, "languages": f.languages,
                        "fields": [{"name": x.name, "label": x.label_for(f.language), "required": x.required}
                                   for x in f.fields]})
        return out

    # ---- calls
    def _gc(self) -> None:
        now = self.clock()
        with self._lock:
            for cid in [c for c, (_, _, t) in self._sessions.items() if now - t[0] > self.ttl]:
                del self._sessions[cid]

    def start_call(self, form_id: str, language: Optional[str] = None) -> Dict[str, Any]:
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

    def turn(self, call_id: str, wav: bytes) -> Dict[str, Any]:
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

    def event(self, call_id: str, name: str) -> Dict[str, Any]:
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
                raise ApiError(401, "missing or wrong token")

        def _route(self, method: str) -> None:
            try:
                self._check_host()
                path = urlparse(self.path).path
                if method == "GET" and path in ("/", "/index.html", "/admin", "/caller", "/forms", "/responses",
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
                if method == "GET" and parts == ["forms"]:
                    return self._json(200, app.list_forms())
                if method == "GET" and parts == ["calls"]:
                    return self._json(200, app.calls.list())
                if method == "GET" and len(parts) == 2 and parts[0] == "call":
                    return self._json(200, app.record(parts[1]))
                if method == "POST" and parts == ["call", "start"]:
                    d = self._json_body()
                    return self._json(200, app.start_call(str(d.get("form_id", "")), d.get("language")))
                if method == "POST" and len(parts) == 3 and parts[0] == "call" and parts[2] == "turn":
                    return self._json(200, app.turn(parts[1], self._body(MAX_AUDIO)))
                if method == "POST" and len(parts) == 3 and parts[0] == "call" and parts[2] == "event":
                    return self._json(200, app.event(parts[1], str(self._json_body().get("event", ""))))
                if method == "POST" and parts == ["tts"]:
                    d = self._json_body()
                    data = app.synth(str(d.get("text", "")), str(d.get("lang", "en")))
                    return self._send(200, data, "audio/mpeg")
                raise ApiError(404, "not found")
            except ApiError as exc:
                self._json(exc.status, {"error": exc.message})
            except ConnectionError:                       # the browser went away (Windows: ConnectionAbortedError): ignore
                pass
            except Exception:                                              # noqa: BLE001 - never leak a traceback
                log.exception("unhandled error")
                try:
                    self._json(500, {"error": "internal error"})
                except OSError:
                    pass

        def _json_body(self) -> Dict[str, Any]:
            raw = self._body(MAX_JSON)
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
              tts_cache: Optional[str] = None, use_tts: bool = True, transcriber: Optional[Transcriber] = None) -> CallApp:
    from callpage.pipeline import make_transcriber
    tts_fn = None
    if use_tts:
        tts_fn = lambda text, lang: tts_mod.synth(text, lang, tts_cache or os.path.join(os.path.dirname(calls_dir.rstrip("/\\")) or ".", "tts_cache"))
    return CallApp(FormStore(forms_dir), CallStore(calls_dir), transcriber or make_transcriber(preprocess, engine), tts_fn)


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
    a = ap.parse_args(argv)
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
    app = build_app(a.forms, a.calls, a.preprocess, a.engine, use_tts=not a.no_tts, transcriber=transcriber)
    if not app.forms.list():
        print(f"No forms found in {a.forms}. Create one in the Admin page, or copy data/forms/clinic_registration.json there.")
    srv = CallServer(app, a.host, a.port)
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
