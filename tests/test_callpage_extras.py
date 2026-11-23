"""Tests for the launcher, export and form-builder helpers."""
import csv
import io
import os
import sys
import time

import pytest

import extractor
from callpage import export, forms, launcher
from callpage.forms import FormError, FormStore, build_form_dict, slugify
from callpage.server import CallApp, CallServer
from callpage.session import CallSession
from callpage.store import CallStore
from callpage.testing import scripted_transcriber


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    extractor.reset_extractor()


@pytest.mark.parametrize("title,slug", [("Clinic registration", "clinic_registration"), ("  Scheme (2026)!  ", "scheme_2026"),
                                        ("ക്ലിനിക്", "form"), ("", "form"), ("A" * 200, "a" * 64), ("---", "form")])
def test_slugify(title, slug):
    assert slugify(title) == slug


def test_build_form_dict_and_save(tmp_path):
    d = build_form_dict("Clinic registration", ["name", {"name": "age", "required": False}], language="hi", languages=["hi", "en"],
                        settings={"store_audio": True})
    st = FormStore(str(tmp_path))
    f = st.save(d)
    assert f.id == "clinic_registration" and f.language == "hi" and f.fields[1].required is False and f.settings["store_audio"] is True
    assert st.get("clinic_registration").title == "Clinic registration"
    with pytest.raises(FormError):
        st.save(build_form_dict("Empty", []))


def test_export_rows_and_csv(tmp_path):
    store = CallStore(str(tmp_path))
    f = forms.FormSpec.from_dict({"id": "c", "title": "C", "fields": ["name", "aadhaar", {"name": "blood_group", "required": False}],
                                  "settings": {"require_consent": False}})
    s = CallSession(f, store=store, today=None, ex=extractor.Extractor(), call_id="abcdef0123456789")
    s.start(); s.handle_text("name is anil"); s.handle_text("yes"); s.handle_text("2 3 4 5 6 7 8 9 0 1 2 4"); s.handle_text("yes")
    s.handle_text("skip"); s.handle_text("yes")
    rows = export.to_rows(export.records(store, "c"))
    assert rows[0]["name"] == "Anil" and rows[0]["aadhaar"] == "234567890124" and rows[0]["blood_group"] == "(skipped)"
    assert rows[0]["outcome"] == "completed" and rows[0]["answered"] == "3/3"
    masked = export.to_rows(export.records(store, "c"), mask_sensitive=True)
    assert masked[0]["aadhaar"] == "**********24"
    parsed = list(csv.DictReader(io.StringIO(export.to_csv(rows))))
    assert parsed[0]["name"] == "Anil" and "call_id" in parsed[0]
    assert export.records(store, "other") == [] and export.to_csv([]) == "\r\n"


def test_export_skips_unreadable_files(tmp_path):
    store = CallStore(str(tmp_path))
    (tmp_path / "abcdef0123456789.json").write_text("{ broken")
    assert export.records(store) == []


def test_call_url():
    assert launcher.call_url("clinic_registration", "hi") == "http://127.0.0.1:8765/?form=clinic_registration&lang=hi"
    assert launcher.call_url("a b", None, 9000) == "http://127.0.0.1:9000/?form=a+b"


def test_server_running_detects_a_live_server(tmp_path):
    srv = CallServer(CallApp(FormStore(str(tmp_path)), CallStore(str(tmp_path)), scripted_transcriber(["x"])), port=0).start()
    try:
        assert launcher.server_running(srv.port) is True
    finally:
        srv.stop()
    assert launcher.server_running(srv.port) is False and launcher.server_running(1) is False


def test_ensure_server_starts_a_real_background_process(tmp_path):
    forms_dir = tmp_path / "forms"; FormStore(str(forms_dir)).save(build_form_dict("Demo", ["name"]))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    try:
        url = launcher.ensure_server(port, str(forms_dir), str(tmp_path / "calls"), ("--no-tts",), project_root=root, wait_s=15)
        assert url == f"http://127.0.0.1:{port}/" and launcher.server_running(port)
        assert launcher.ensure_server(port, project_root=root) == url                      # already running: no second process
    finally:
        launcher.stop_server()
        time.sleep(0.3)
        for fn in ("callpage_server.log",):
            p = os.path.join(root, fn)
            if os.path.exists(p):
                for _ in range(10):
                    try:
                        os.remove(p)
                        break
                    except PermissionError:
                        time.sleep(0.1)
    assert not launcher.server_running(port)
