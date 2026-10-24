"""The example settings file and the repository must never contain a real key."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$")
# A Z.ai key: 32 hex characters, a dot, 16 letters or digits. A Groq key: gsk_ and 20 or more characters.
KEY_SHAPE = re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9]{16}\b|\bgsk_[A-Za-z0-9]{20,}")
MAX_BYTES = 1_000_000


def secret_assignments(text):
    """(name, value) for every NAME=value line whose name ends in KEY, TOKEN, SECRET or PASSWORD."""
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if SECRET_NAME.search(name.strip()):
            found.append((name.strip(), value.strip()))
    return found


def key_like_strings(text):
    return KEY_SHAPE.findall(text)


# ---------- the example file ----------

def test_the_example_file_has_no_secret_values():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    names = [name for name, _ in secret_assignments(text)]
    assert "GROQ_API_KEY" in names and "ZAI_API_KEY" in names
    filled = [name for name, value in secret_assignments(text) if value]
    assert filled == [], f"these variables have a value in .env.example: {filled}"


def test_the_rule_catches_a_filled_secret_and_ignores_the_rest():
    sample = "# comment\nA_KEY=abc\nB_TOKEN=\nC_MODEL=x\nD_SECRET = s\nE_PASSWORD=\n"
    assert secret_assignments(sample) == [("A_KEY", "abc"), ("B_TOKEN", ""), ("D_SECRET", "s"), ("E_PASSWORD", "")]


# ---------- the whole repository ----------

def test_the_key_shapes_are_recognised():
    assert key_like_strings("x " + "a" * 32 + "." + "B" * 16 + " y")
    assert key_like_strings("gsk_" + "a" * 20)
    assert not key_like_strings("gsk_short and " + "a" * 32 + "." + "b" * 3)


def tracked_text_files():
    try:
        listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available here")
    for name in filter(None, listed):
        path = ROOT / name
        if name.startswith("tests/fixtures") or not path.is_file() or path.stat().st_size > MAX_BYTES:
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # a binary file


def test_no_key_shaped_string_in_any_tracked_or_new_text_file():
    hits = [name for name, text in tracked_text_files() if key_like_strings(text)]
    assert hits == [], f"key-shaped strings found in: {hits}"
