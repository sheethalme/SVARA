"""Make sure the sample clips exist (they are binary files that may be missing in a clone)."""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture(scope="session", autouse=True)
def _ensure_sample_audio():
    import make_sample_data
    make_sample_data.build(only_missing=True)
