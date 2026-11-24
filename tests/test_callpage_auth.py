"""Accounts (user name + password, no e-mail), roles, and private forms/calls, tested over real HTTP."""
import json
import os
import urllib.error
import urllib.request

import pytest

from callpage.server import CallServer, build_app
from callpage.testing import scripted_transcriber

FORM = {"id": "clinic", "title": "Clinic", "language": "en", "fields": [{"name": "name"}]}


def make_server(tmp_path):
    forms = tmp_path / "data" / "forms"
    forms.mkdir(parents=True, exist_ok=True)
    (forms / "clinic.json").write_text(json.dumps(FORM), encoding="utf-8")
    app = build_app(str(forms), str(tmp_path / "data" / "calls"), use_tts=False, transcriber=scripted_transcriber(["yes"]), accounts=True)
    return CallServer(app, "127.0.0.1", 0).start()


@pytest.fixture(autouse=True)
def _keep_the_field_registry_clean():
    """Publishing a PDF form registers new answer fields globally; undo that so other tests do not see them."""
    import extractor
    fields, aliases = dict(extractor.FIELDS), dict(extractor._ALIASES)
    yield
    extractor.FIELDS.clear(); extractor.FIELDS.update(fields)
    extractor._ALIASES.clear(); extractor._ALIASES.update(aliases)


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SVARA_ADMIN_USERNAME", "boss")
    monkeypatch.setenv("SVARA_ADMIN_PASSWORD", "correct horse battery")
    server = make_server(tmp_path)
    yield server, tmp_path
    server.stop()


def call(server, path, body=None, session=None, method=None):
    req = urllib.request.Request(server.url.rstrip("/") + path, method=method or ("POST" if body is not None else "GET"),
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("X-Call-Token", server.token)
    req.add_header("Content-Type", "application/json")
    if session:
        req.add_header("X-Session", session)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def login(server, username, password, role):
    st, body = call(server, "/api/auth/login", {"username": username, "password": password, "role": role})
    assert st == 200, body
    return body["token"]


def make_caller(server, username, password="longenough1"):
    st, body = call(server, "/api/auth/signup", {"name": "Caller " + username, "username": username, "password": password})
    assert st == 200, body
    return login(server, username, password, "user")


def test_api_needs_a_session(srv):
    server, _ = srv
    assert call(server, "/api/forms")[0] == 401


def test_create_account_then_sign_in(srv):
    server, _ = srv
    st, body = call(server, "/api/auth/signup", {"name": "Anil Kumar", "username": "Anil", "password": "secret123"})
    assert st == 200 and body["user"] == {"username": "anil", "name": "Anil Kumar", "role": "user"}
    assert "token" not in body                                                  # creating an account does not sign you in
    st, body = call(server, "/api/auth/login", {"username": "ANIL", "password": "secret123", "role": "user"})
    assert st == 200 and body["token"]                                          # user names ignore capitals
    assert call(server, "/api/forms", session=body["token"])[0] == 200


def test_only_registered_accounts_can_sign_in(srv):
    server, _ = srv
    assert call(server, "/api/auth/login", {"username": "nobody", "password": "whatever123", "role": "user"})[0] == 401
    make_caller(server, "realuser")
    assert call(server, "/api/auth/login", {"username": "realuser", "password": "wrong-password", "role": "user"})[0] == 401


def test_bad_inputs_duplicates_and_lockout(srv):
    server, _ = srv
    assert call(server, "/api/auth/signup", {"name": "A", "username": "ab", "password": "x"})[0] == 400
    assert call(server, "/api/auth/signup", {"name": "Anil", "username": "a@b.c", "password": "longenough1"})[0] == 400   # no @ in a user name
    assert call(server, "/api/auth/signup", {"name": "Anil", "username": "anil2", "password": "short"})[0] == 400
    make_caller(server, "taken")
    assert call(server, "/api/auth/signup", {"name": "Other", "username": "TAKEN", "password": "longenough1"})[0] == 409
    for _ in range(5):
        assert call(server, "/api/auth/login", {"username": "boss", "password": "wrong"})[0] == 401
    assert call(server, "/api/auth/login", {"username": "boss", "password": "correct horse battery"})[0] == 429


def test_roles_are_enforced(srv):
    server, _ = srv
    admin = login(server, "boss", "correct horse battery", "admin")
    caller = make_caller(server, "caller1")
    assert call(server, "/api/calls", session=admin)[0] == 200
    assert call(server, "/api/calls", session=caller)[0] == 403
    assert call(server, "/api/admin/settings", session=caller)[0] == 403
    assert call(server, "/api/admin/forms/save", {"title": "x", "fields": []}, session=caller)[0] == 403
    assert call(server, "/api/auth/login", {"username": "boss", "password": "correct horse battery", "role": "user"})[0] == 401   # wrong door


def test_callers_cannot_touch_each_others_calls(srv):
    server, _ = srv
    one, two = make_caller(server, "caller1"), make_caller(server, "caller2")
    st, started = call(server, "/api/call/start", {"form_id": "clinic", "language": "en"}, session=one)
    assert st == 200
    cid = started["call_id"]
    assert call(server, f"/api/call/{cid}/event", {"event": "no_speech"}, session=two)[0] == 404
    assert call(server, f"/api/call/{cid}/filled.pdf", session=two)[0] == 404
    assert call(server, f"/api/call/{cid}/event", {"event": "hangup"}, session=one)[0] == 200


def test_logout_ends_the_session(srv):
    server, _ = srv
    token = login(server, "boss", "correct horse battery", "admin")
    assert call(server, "/api/auth/me", session=token)[0] == 200
    call(server, "/api/auth/logout", {}, session=token)
    assert call(server, "/api/auth/me", session=token)[0] == 401


def test_a_callers_uploaded_form_is_private_and_can_be_filled(srv):
    server, _ = srv
    pdf = open(os.path.join(os.path.dirname(__file__), "..", "sample_form_admission.pdf"), "rb").read()
    one, two = make_caller(server, "caller1"), make_caller(server, "caller2")
    admin = login(server, "boss", "correct horse battery", "admin")

    def upload(session):
        req = urllib.request.Request(server.url + "api/admin/forms/analyze", data=pdf, method="POST")
        for k, v in {"X-Call-Token": server.token, "X-Session": session, "Content-Type": "application/pdf", "X-Filename": "f.pdf"}.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    analysed = upload(one)
    assert analysed["fields"]
    assert call(server, "/api/admin/forms/publish", {"template_id": analysed["template_id"]}, session=two)[0] == 403   # not theirs
    st, saved = call(server, "/api/admin/forms/publish", {"template_id": analysed["template_id"]}, session=one)
    assert st == 200
    mine = {f["id"]: f for f in call(server, "/api/forms", session=one)[1]}
    assert mine[saved["id"]]["mine"] is True and mine[saved["id"]]["has_pdf"] is True
    assert saved["id"] not in {f["id"] for f in call(server, "/api/forms", session=two)[1]}       # private to the owner
    assert saved["id"] not in {f["id"] for f in call(server, "/api/forms", session=admin)[1]}
    assert call(server, "/api/call/start", {"form_id": saved["id"]}, session=two)[0] == 404
    st, started = call(server, "/api/call/start", {"form_id": saved["id"], "language": "en"}, session=one)
    assert st == 200
    call(server, f"/api/call/{started['call_id']}/event", {"event": "hangup"}, session=one)
    req = urllib.request.Request(server.url + f"api/call/{started['call_id']}/filled.pdf")
    req.add_header("X-Call-Token", server.token)
    req.add_header("X-Session", one)
    with urllib.request.urlopen(req) as r:
        assert r.status == 200 and r.read(4) == b"%PDF"
    assert call(server, "/api/admin/forms/delete", {"id": saved["id"]}, session=two)[0] == 403


def test_first_admin_is_created_once_on_the_admin_page(tmp_path, monkeypatch):
    for k in ("SVARA_ADMIN_USERNAME", "SVARA_ADMIN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    server = make_server(tmp_path)
    try:
        assert call(server, "/api/auth/status")[1] == {"admin_exists": False}
        assert call(server, "/api/auth/login", {"username": "admin", "password": "svara-admin", "role": "admin"})[0] == 401   # no built-in account
        body = {"name": "Me", "username": "owner", "password": "longenough1", "role": "admin"}
        assert call(server, "/api/auth/signup", body)[0] == 200
        assert call(server, "/api/auth/status")[1] == {"admin_exists": True}
        assert login(server, "owner", "longenough1", "admin")
        assert call(server, "/api/auth/signup", {**body, "username": "second"})[0] == 403           # only one admin sign-up
    finally:
        server.stop()


def test_accounts_survive_a_restart(tmp_path, monkeypatch):
    for k in ("SVARA_ADMIN_USERNAME", "SVARA_ADMIN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    first = make_server(tmp_path)
    try:
        make_caller(first, "keeper", "stays-put-1")
    finally:
        first.stop()
    second = make_server(tmp_path)
    try:
        assert login(second, "keeper", "stays-put-1", "user")
    finally:
        second.stop()


def test_old_accounts_are_migrated(tmp_path, monkeypatch):
    """Accounts made by the previous (e-mail + code) version keep working; unconfirmed ones and the default admin go."""
    from callpage.auth import UserStore, _hash
    monkeypatch.delenv("SVARA_ADMIN_USERNAME", raising=False)
    salt = bytes.fromhex("aa" * 16)
    old = {"me@example.org": {"email": "me@example.org", "name": "Me", "username": "me.old", "role": "user", "verified": True,
                              "salt": salt.hex(), "hash": _hash("old-password", salt)},
           "half@example.org": {"email": "half@example.org", "name": "Half", "username": "half", "role": "user", "verified": False,
                                "salt": salt.hex(), "hash": _hash("x" * 8, salt)},
           "admin@svara.local": {"email": "admin@svara.local", "name": "Administrator", "username": "admin", "role": "admin",
                                 "verified": True, "salt": salt.hex(), "hash": _hash("svara-admin", salt)}}
    path = tmp_path / "users.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    store = UserStore(str(path))
    assert store.login("me.old", "old-password", "user")["username"] == "me.old"
    assert store.renamed == {"me@example.org": "me.old"}
    assert not store.admin_exists()
    with pytest.raises(Exception):
        store.login("half", "xxxxxxxx")


# ------------------------------------------------------------------ per-form access codes
def set_code(server, admin, form_id="clinic", action="generate", code=None):
    st, body = call(server, "/api/admin/forms/access", {"id": form_id, "action": action, **({"code": code} if code else {})}, session=admin)
    assert st == 200, body
    return body["access_code"]


def visible(server, session):
    return {f["id"] for f in call(server, "/api/forms", session=session)[1]}


def enter(server, session, code):
    return call(server, "/api/forms/unlock-by-code", {"code": code}, session=session)


def test_a_form_with_a_code_is_invisible_until_the_code_is_entered(srv):
    server, _ = srv
    admin = login(server, "boss", "correct horse battery", "admin")
    code = set_code(server, admin)
    assert len(code) == 6
    one, two = make_caller(server, "caller1"), make_caller(server, "caller2")

    assert visible(server, one) == set() and visible(server, two) == set()                  # not even listed
    assert "clinic" not in json.dumps(call(server, "/api/forms", session=one)[1])
    assert code not in json.dumps(call(server, "/api/forms", session=one)[1])               # callers never see a code
    assert visible(server, admin) == {"clinic"} and {f["id"]: f for f in call(server, "/api/forms", session=admin)[1]}["clinic"]["access_code"] == code
    assert call(server, "/api/call/start", {"form_id": "clinic", "language": "en"}, session=one)[0] == 404   # looks like it does not exist
    assert enter(server, one, "WRONG1")[0] == 403 and visible(server, one) == set()
    st, body = enter(server, one, code.lower()[:3] + "-" + code.lower()[3:])                # case and dashes are ignored
    assert st == 200 and body["forms"] == [{"id": "clinic", "title": "Clinic"}]
    assert visible(server, one) == {"clinic"}
    assert call(server, "/api/call/start", {"form_id": "clinic", "language": "en"}, session=one)[0] == 200
    assert visible(server, two) == set()                                                    # the other caller still has no access
    assert call(server, "/api/call/start", {"form_id": "clinic"}, session=two)[0] == 404


def test_changing_the_code_revokes_access_and_removing_it_opens_the_form(srv):
    server, _ = srv
    admin = login(server, "boss", "correct horse battery", "admin")
    first = set_code(server, admin)
    one = make_caller(server, "caller1")
    assert enter(server, one, first)[0] == 200 and visible(server, one) == {"clinic"}
    assert set_code(server, admin, action="set", code="open-sesame") == "OPENSESAME"
    assert visible(server, one) == set()                                                    # the old unlock no longer counts
    assert enter(server, one, first)[0] == 403
    assert enter(server, one, "OPENSESAME")[0] == 200 and visible(server, one) == {"clinic"}
    set_code(server, admin, action="clear")
    two = make_caller(server, "caller2")
    assert visible(server, two) == {"clinic"}                                               # an admin who removes the code opens the form
    assert call(server, "/api/admin/forms/access", {"id": "clinic", "action": "set", "code": "ab"}, session=admin)[0] == 400


def test_only_the_admin_manages_codes_and_guessing_is_limited(srv):
    server, _ = srv
    admin = login(server, "boss", "correct horse battery", "admin")
    set_code(server, admin)
    one = make_caller(server, "caller1")
    assert call(server, "/api/admin/forms/access", {"id": "clinic", "action": "clear"}, session=one)[0] == 403
    assert enter(server, one, "")[0] == 403
    for _ in range(4):
        assert enter(server, one, "NOPE99")[0] == 403
    assert enter(server, one, "NOPE99")[0] == 429                                           # five wrong tries: wait a minute
    assert visible(server, one) == set()


def test_forms_the_admin_creates_get_a_code_automatically(srv):
    server, _ = srv
    admin = login(server, "boss", "correct horse battery", "admin")
    st, saved = call(server, "/api/admin/forms/save", {"title": "Eye test", "languages": ["en"],
                                                       "fields": [{"name": "custom_1", "label": "Your name", "kind": "name", "required": True}]}, session=admin)
    assert st == 200 and len(saved["access_code"]) == 6
    caller = make_caller(server, "caller1")
    assert saved["id"] not in visible(server, caller)
    assert call(server, "/api/call/start", {"form_id": saved["id"]}, session=caller)[0] == 404
    assert enter(server, caller, saved["access_code"])[0] == 200 and saved["id"] in visible(server, caller)


def test_a_callers_own_private_form_never_asks_for_a_code(srv):
    server, _ = srv
    pdf = open(os.path.join(os.path.dirname(__file__), "..", "sample_form_admission.pdf"), "rb").read()
    one = make_caller(server, "caller1")
    req = urllib.request.Request(server.url + "api/admin/forms/analyze", data=pdf, method="POST")
    for k, v in {"X-Call-Token": server.token, "X-Session": one, "Content-Type": "application/pdf", "X-Filename": "f.pdf"}.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req) as r:
        analysed = json.loads(r.read())
    st, saved = call(server, "/api/admin/forms/publish", {"template_id": analysed["template_id"]}, session=one)
    assert st == 200 and "access_code" not in saved
    assert saved["id"] in visible(server, one)
    assert call(server, "/api/call/start", {"form_id": saved["id"], "language": "en"}, session=one)[0] == 200


def test_the_old_per_form_unlock_endpoint_is_gone(srv):
    server, _ = srv
    admin = login(server, "boss", "correct horse battery", "admin")
    code = set_code(server, admin)
    one = make_caller(server, "caller1")
    assert call(server, "/api/forms/unlock", {"form_id": "clinic", "code": code}, session=one)[0] == 404
