"""Keep every raw model answer on disk, so a run can be repeated or resumed without new API calls."""
import hashlib
import json
from pathlib import Path


def make_key(model, prompt_version, text):
    return hashlib.sha256(f"{model}|{prompt_version}|{text}".encode()).hexdigest()


class JsonlCache:
    """One JSON object per line: key, odi_no, raw answer, token counts and latency. The complaint text is not stored."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["key"]] = record

    def get(self, key):
        return self.records.get(key)

    def put(self, record):
        self.records[record["key"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
