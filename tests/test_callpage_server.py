"""Tests for the call server: the application layer, the HTTP layer and its security checks. No browser, no network."""
import http.client
import io
import json
import time
import wave

import pytest

import extractor
from callpage import tts as tts_mod
from callpage.forms import FormStore
from callpage.server import CallApp, CallServer, ApiError
from callpage.session import TranscribeResult
from callpage.store import CallStore
from callpage.testing import scripted_transcriber


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "dotenv", None)
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "SVARA_EXTRACTOR_LLM"):
        monkeypatch.delenv(k, raising=False)
    extractor.reset_extractor()


def make_wav(seconds=0.4, sr=16000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(b"\x01\x02" * int(sr * seconds))
    return b.getvalue()


HAPPY = ["yes", "my name is anil kumar", "yes", "I am twenty five years old", "yes", "male", "yes", "9876543210", "yes", "skip", "yes"]


@pytest.fixture
def stack(tmp_path):
    forms = FormStore(str(tmp_path / "forms"))
    forms.save({"id": "clinic", "title": "Clinic registration", "language": "en", "settings": {"store_audio": True},
                "fields": ["name", "age", "gender", "phone", {"name": "blood_group", "required": False}]})
    forms.save({"id": "hindi_form", "title": "Hindi form", "language": "hi", "languages": ["hi", "en"], "fields": ["name"]})
    calls = CallStore(str(tmp_path / "calls"))
    app = CallApp(forms, calls, scripted_transcriber(HAPPY), tts_fn=lambda t, l: b"ID3" + t.encode()[:20])
    srv = CallServer(app, port=0).start()
    yield srv, app, calls
    srv.stop()


def req(srv, method, path, body=None, headers=None, token=True, host=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    h = dict(headers or {})
    if token:
        h["X-Call-Token"] = srv.token
    if host is not None:
        h["Host"] = host
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode(); h.setdefault("Content-Type", "application/json")
    c.request(method, path, body=body, headers=h)
    r = c.getresponse(); data = r.read(); c.close()
    return r.status, data, dict(r.getheaders())


def jreq(*a, **k):
    s, d, _ = req(*a, **k)
    return s, (json.loads(d) if d and d[:1] in b"{[" else d)


# ====================================================================== the full call over HTTP
def test_full_call_over_http_and_stored(stack):
    srv, app, calls = stack
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"})
    assert s == 200 and first["turn"]["phase"] == "consent" and "Clinic registration" in first["turn"]["say"]
    cid = first["call_id"]
    turn = first["turn"]; n = 0
    while not turn["done"]:
        s, r = jreq(srv, "POST", f"/api/call/{cid}/turn", make_wav(), {"Content-Type": "audio/wav"})
        assert s == 200; turn = r["turn"]; n += 1
        assert n < 30
    assert turn["outcome"] == "completed" and "Your form has been saved" in turn["say"]
    s, rec = jreq(srv, "GET", f"/api/call/{cid}")
    assert s == 200 and {f["name"]: f["value"] for f in rec["fields"]} == {
        "name": "Anil Kumar", "age": "25", "gender": "Male", "phone": "9876543210", "blood_group": None}
    assert rec["consent"] is True and rec["outcome"] == "completed"
    assert calls.save_audio.__self__ is calls and len(list((__import__("pathlib").Path(calls.dir) / cid).glob("*.wav"))) >= 5   # store_audio form + consent
    s, lst = jreq(srv, "GET", "/api/calls")
    assert [c["call_id"] for c in lst] == [cid]
    s, r = jreq(srv, "POST", f"/api/call/{cid}/turn", make_wav(), {"Content-Type": "audio/wav"})
    assert s == 404                                                         # a finished call is gone from memory


def test_audio_is_not_stored_without_consent_or_setting(stack):
    srv, app, calls = stack
    app.transcriber = scripted_transcriber(["no"])                          # consent declined on a form that stores audio
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"})
    jreq(srv, "POST", f"/api/call/{first['call_id']}/turn", make_wav(), {"Content-Type": "audio/wav"})
    import pathlib
    assert not (pathlib.Path(calls.dir) / first["call_id"]).exists()
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "hindi_form", "language": "hi"})   # form without store_audio
    app.transcriber = scripted_transcriber(["हाँ"])
    jreq(srv, "POST", f"/api/call/{first['call_id']}/turn", make_wav(), {"Content-Type": "audio/wav"})
    assert not (pathlib.Path(calls.dir) / first["call_id"]).exists()


def test_events_no_speech_and_hangup(stack):
    srv, app, calls = stack
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"}); cid = first["call_id"]
    s, r = jreq(srv, "POST", f"/api/call/{cid}/event", {"event": "no_speech"})
    assert s == 200 and r["turn"]["say"].startswith("I could not hear you.")
    s, r = jreq(srv, "POST", f"/api/call/{cid}/event", {"event": "hangup"})
    assert r["turn"]["done"] and r["turn"]["outcome"] == "hangup"
    assert jreq(srv, "GET", f"/api/call/{cid}")[1]["outcome"] == "hangup"
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"})
    assert jreq(srv, "POST", f"/api/call/{first['call_id']}/event", {"event": "dance"})[0] == 400


def test_language_choice(stack):
    srv, app, calls = stack
    s, r = jreq(srv, "POST", "/api/call/start", {"form_id": "hindi_form"})
    assert r["turn"]["lang"] == "hi" and "नमस्ते" in r["turn"]["say"]
    s, r = jreq(srv, "POST", "/api/call/start", {"form_id": "hindi_form", "language": "en"})
    assert r["turn"]["lang"] == "en"
    s, r = jreq(srv, "POST", "/api/call/start", {"form_id": "hindi_form", "language": "kn"})
    assert s == 400 and "not offered" in r["error"]


def test_forms_endpoint(stack):
    srv, *_ = stack
    s, forms = jreq(srv, "GET", "/api/forms")
    f = next(x for x in forms if x["id"] == "clinic")
    assert s == 200 and f["title"] == "Clinic registration" and [x["name"] for x in f["fields"]][:2] == ["name", "age"]
    assert f["fields"][4]["required"] is False


def test_tts_endpoint_and_fallback(stack):
    srv, app, calls = stack
    s, data, h = req(srv, "POST", "/api/tts", {"text": "Hello there", "lang": "en"})
    assert s == 200 and h["Content-Type"] == "audio/mpeg" and data.startswith(b"ID3")
    app.tts_fn = lambda t, l: (_ for _ in ()).throw(tts_mod.TTSUnavailable("no internet"))
    s, r = jreq(srv, "POST", "/api/tts", {"text": "Hello", "lang": "en"})
    assert s == 503 and "no internet" in r["error"]                         # the page then uses the browser voice
    app.tts_fn = None
    assert jreq(srv, "POST", "/api/tts", {"text": "Hello", "lang": "en"})[0] == 503


# ====================================================================== security and robustness
def test_every_api_call_needs_the_token(stack):
    srv, *_ = stack
    for method, path, body in [("GET", "/api/forms", None), ("GET", "/api/calls", None), ("POST", "/api/call/start", {"form_id": "clinic"}),
                               ("GET", "/api/call/abcdef0123456789", None), ("POST", "/api/tts", {"text": "x"})]:
        s, r = jreq(srv, method, path, body, token=False)
        assert s == 401, path
        s, r = jreq(srv, method, path, body, headers={"X-Call-Token": "wrong"}, token=False)
        assert s == 401, path


def test_host_header_must_be_local(stack):
    srv, *_ = stack
    for host in ("evil.example.com", f"evil.com:{srv.port}", "localhost:1", "192.168.1.5:8765", ""):
        assert req(srv, "GET", "/", token=False, host=host)[0] == 403, host
    assert req(srv, "GET", "/", token=False, host=f"localhost:{srv.port}")[0] == 200
    assert req(srv, "GET", "/api/forms", host=f"localhost:{srv.port}")[0] == 200


def test_page_contains_the_token_and_security_headers(stack):
    srv, *_ = stack
    s, body, h = req(srv, "GET", "/", token=False)
    assert s == 200 and srv.token.encode() in body and b"__TOKEN__" not in body
    assert "no-store" in h["Cache-Control"] and h["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in h["Content-Security-Policy"] and "frame-ancestors" not in h["Content-Security-Policy"]
    assert req(srv, "GET", "/healthz", token=False)[1] == b"ok"


def test_size_limits_and_bad_requests(stack):
    srv, *_ = stack
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"}); cid = first["call_id"]
    try:                                       # the server refuses before reading 7 MB: a 413, or it simply hangs up
        assert jreq(srv, "POST", f"/api/call/{cid}/turn", b"x" * (7 * 1024 * 1024), {"Content-Type": "audio/wav"})[0] == 413
    except ConnectionError:                    # Linux: broken pipe / reset; Windows: ConnectionAbortedError (all are ConnectionError)
        pass
    assert jreq(srv, "POST", f"/api/call/{cid}/turn", b"tiny", {"Content-Type": "audio/wav"})[0] == 400
    assert jreq(srv, "POST", "/api/call/start", b"x" * 70000, {"Content-Type": "application/json"})[0] == 413
    assert jreq(srv, "POST", "/api/call/start", b"{not json", {"Content-Type": "application/json"})[0] == 400
    assert jreq(srv, "POST", "/api/call/start", b"[1,2]", {"Content-Type": "application/json"})[0] == 400
    assert jreq(srv, "POST", "/api/call/start", {"form_id": "nope"})[0] == 404
    assert jreq(srv, "POST", "/api/call/start", {"form_id": "../etc/passwd"})[0] == 404
    assert jreq(srv, "POST", "/api/call/start", {})[0] == 404


def test_unknown_routes_and_methods(stack):
    srv, *_ = stack
    assert jreq(srv, "GET", "/nothing")[0] == 404 and jreq(srv, "GET", "/api/nothing")[0] == 404
    assert jreq(srv, "POST", "/api/forms", {})[0] == 404 and jreq(srv, "GET", "/api/call/start")[0] == 404


def test_call_ids_cannot_reach_other_files(stack, tmp_path):
    srv, *_ = stack
    for bad in ("../forms/clinic", "..%2fforms%2fclinic", "abc", "ABCDEF0123456789", "abcdef0123456789/../x"):
        assert jreq(srv, "GET", f"/api/call/{bad}")[0] == 404, bad
        assert jreq(srv, "POST", f"/api/call/{bad}/event", {"event": "hangup"})[0] == 404, bad


def test_internal_errors_do_not_leak_details(stack):
    srv, app, calls = stack
    app.forms.list = lambda: (_ for _ in ()).throw(RuntimeError("SECRET /home/user/path"))
    s, data, _ = req(srv, "GET", "/api/forms")
    assert s == 500 and b"SECRET" not in data and json.loads(data) == {"error": "internal error"}


def test_a_failing_recogniser_does_not_break_the_http_call(stack):
    srv, app, calls = stack
    app.transcriber = lambda w, l, h: TranscribeResult(error="network error")
    s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"}); cid = first["call_id"]
    s, r = jreq(srv, "POST", f"/api/call/{cid}/turn", make_wav(), {"Content-Type": "audio/wav"})
    assert s == 200 and "something went wrong" in r["turn"]["say"] and not r["turn"]["done"]


def test_idle_sessions_are_dropped(tmp_path):
    forms = FormStore(str(tmp_path / "f")); forms.save({"id": "c", "title": "C", "fields": ["name"]})
    now = [1000.0]
    app = CallApp(forms, CallStore(str(tmp_path / "c")), scripted_transcriber(["yes"]), session_ttl_s=60, clock=lambda: now[0])
    a = app.start_call("c")["call_id"]
    now[0] += 120
    app.start_call("c")                                                      # starting a call cleans up idle ones
    with pytest.raises(ApiError) as e:
        app.turn(a, make_wav())
    assert e.value.status == 404
    assert CallStore(str(tmp_path / "c")).load(a)["outcome"] == "in_progress"        # its answers were saved all along


def test_concurrent_calls_do_not_mix(stack):
    import threading
    srv, app, calls = stack
    results = {}

    def one(i):
        app.transcriber = lambda w, l, h: TranscribeResult(text="yes")
        s, first = jreq(srv, "POST", "/api/call/start", {"form_id": "clinic"})
        s, r = jreq(srv, "POST", f"/api/call/{first['call_id']}/turn", make_wav(), {"Content-Type": "audio/wav"})
        results[i] = (first["call_id"], r["turn"]["phase"])
    ts = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len({v[0] for v in results.values()}) == 8 and all(v[1] == "ask" for v in results.values())


def test_server_port_zero_and_stop():
    srv = CallServer(CallApp(FormStore("/nonexistent"), CallStore("/nonexistent"), scripted_transcriber(["x"])), port=0).start()
    assert srv.port > 0 and req(srv, "GET", "/healthz", token=False)[0] == 200
    srv.stop()
    with pytest.raises(OSError):
        req(srv, "GET", "/healthz", token=False)
