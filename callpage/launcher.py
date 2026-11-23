"""
launcher.py -- start the call server from another program (e.g. the Streamlit Admin / "Fill Form by Voice" pages)
===============================================================================================================

    from callpage.launcher import ensure_server, call_url
    ensure_server()                                    # starts `python -m callpage` in the background if needed
    st.link_button("Start voice call", call_url("clinic_registration", "hi"))   # opens the call page in a new tab

The call page is a separate web page (http://127.0.0.1:8765) because a call needs the microphone and the speaker
to work continuously, which Streamlit's re-run model cannot do.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence
from urllib.parse import urlencode

DEFAULT_PORT = 8765
_proc: Optional[subprocess.Popen] = None


def base_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}/"


def server_running(port: int = DEFAULT_PORT, timeout: float = 0.6) -> bool:
    try:
        with urllib.request.urlopen(base_url(port) + "healthz", timeout=timeout) as r:
            return r.read() == b"ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def ensure_server(port: int = DEFAULT_PORT, forms_dir: str = os.path.join("data", "forms"),
                  calls_dir: str = os.path.join("data", "calls"), extra_args: Sequence[str] = (),
                  project_root: Optional[str] = None, wait_s: float = 8.0) -> str:
    """Return the call page URL, starting the server (a background process) if it is not running yet."""
    global _proc
    if server_running(port):
        return base_url(port)
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "callpage_server.log"), "ab") as log:
        _proc = subprocess.Popen([sys.executable, "-m", "callpage", "--port", str(port), "--forms", forms_dir, "--calls", calls_dir,
                                  *extra_args], cwd=root, stdout=log, stderr=log)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if server_running(port):
            return base_url(port)
        if _proc.poll() is not None:
            raise RuntimeError(f"the call server stopped at start-up; see {os.path.join(root, 'callpage_server.log')}")
        time.sleep(0.2)
    raise RuntimeError("the call server did not start in time; see callpage_server.log")


def stop_server() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.wait(timeout=3)
    _proc = None


def call_url(form_id: str, language: Optional[str] = None, port: int = DEFAULT_PORT) -> str:
    q = {"form": form_id}
    if language:
        q["lang"] = language
    return base_url(port) + "?" + urlencode(q)
