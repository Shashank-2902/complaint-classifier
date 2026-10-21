import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prepare import load_raw, run  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "raw_sample.tsv"


@pytest.fixture(scope="session")
def raw_sample():
    """444 real complaint rows in the raw NHTSA format (see scripts/make_test_fixture.py)."""
    return load_raw.read_raw_tsv(FIXTURE)


@pytest.fixture(scope="session")
def built(raw_sample):
    """The prepared dataset and its report, made from the fixture."""
    return run.build(raw_sample, sample_size=10_000, top_k=14)


@pytest.fixture
def prepared_dir(built, tmp_path):
    """The prepared dataset saved on disk, the way src.prepare.run saves it."""
    frame, report = built
    processed, results = tmp_path / "processed", tmp_path / "results"
    run.save(frame, report, processed, results)
    return processed
