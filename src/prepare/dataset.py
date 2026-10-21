"""Load the prepared data. The test set is only handed out through one guarded function."""
import json
from datetime import datetime, timezone

import pandas as pd

from src import config

DATASET_FILE = "dataset.csv.gz"
TEST_LOG = config.RESULTS_DIR / "test_access_log.jsonl"


class TestSetAlreadyUsed(Exception):
    """The test set of this split was already used for this system."""


def load_all(processed_dir=config.PROCESSED_DIR):
    return pd.read_csv(processed_dir / DATASET_FILE, dtype={"odi_no": str})


def load_split(scheme, processed_dir=config.PROCESSED_DIR):
    """Return {'train': ..., 'val': ...}. The test rows are NOT included, so tuning cannot see them."""
    frame = load_all(processed_dir)
    column = f"split_{scheme}"
    return {part: frame[frame[column] == part].reset_index(drop=True) for part in ["train", "val"]}


def class_list(processed_dir=config.PROCESSED_DIR):
    frame = load_all(processed_dir)
    named = sorted(c for c in frame["label"].unique() if c != config.OTHER)
    return named + [config.OTHER]


def used_tests(log_path=TEST_LOG):
    """All (system, scheme) pairs that opened a test set. Every file next to the log with the name
    test_access_log*.jsonl counts, so the log that comes back from the Colab run is honoured too."""
    used = set()
    for path in sorted(log_path.parent.glob("test_access_log*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                used.add((entry["system"], entry["scheme"]))
    return used


def load_test_once(scheme, system, processed_dir=config.PROCESSED_DIR, log_path=TEST_LOG, force=False):
    """Return the test rows of a split. Each system may open each test set only once.

    The access is written to a log (results/test_access_log.jsonl), which is committed with the
    results, so anybody can see that the test set was not used for tuning.
    """
    if (system, scheme) in used_tests(log_path) and not force:
        raise TestSetAlreadyUsed(f"{system} already used the {scheme} test set. Use --force to repeat it.")

    frame = load_all(processed_dir)
    test = frame[frame[f"split_{scheme}"] == "test"].reset_index(drop=True)
    log_access(scheme, system, len(test), force, log_path)
    return test


def log_access(scheme, system, rows, force, log_path=TEST_LOG):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"system": system, "scheme": scheme, "rows": rows, "forced": force,
             "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_test_items_once(schemes, system, ids, processed_dir=config.PROCESSED_DIR, log_path=TEST_LOG, force=False):
    """Open only some test items (given by id) for a system that is run once for several splits.

    Every id must be a test item in every one of the schemes. One log entry is written per scheme.
    """
    frame = load_all(processed_dir)
    chosen = frame[frame["odi_no"].isin(set(ids))]
    for scheme in schemes:
        if (system, scheme) in used_tests(log_path) and not force:
            raise TestSetAlreadyUsed(f"{system} already used the {scheme} test set. Use --force to repeat it.")
        if not (chosen[f"split_{scheme}"] == "test").all():
            raise ValueError(f"some of the items are not test items of the {scheme} split")
    for scheme in schemes:
        log_access(scheme, system, len(chosen), force, log_path)
    return chosen.reset_index(drop=True)
