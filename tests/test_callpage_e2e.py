"""Runs the real-browser end-to-end scenarios (Chromium + a simulated microphone) when Node + Playwright are available.
Skipped on machines without them (most laptops); the scenarios live in tests/e2e/call_e2e.cjs and can be run by hand."""
import os
import shutil
import subprocess

import pytest

E2E = os.path.join(os.path.dirname(__file__), "e2e", "call_e2e.cjs")


def _playwright_ok():
    if not shutil.which("node"):
        return False
    r = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True, timeout=30)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _playwright_ok(), reason="needs Node.js + Playwright with Chromium (dev only)")


@pytest.mark.parametrize("scenario", ["happy", "correction", "silent"])
def test_browser_call(scenario):
    r = subprocess.run(["node", E2E, scenario], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
