"""The fixed test subset used for the Groq row, and for the same-items comparison of all systems.

Groq's free plan allows only a limited number of tokens per day, so the zero-shot model is run once on
1000 items. These items are chosen from the test sets of BOTH splits (the overlap of the two), so the
same Groq answers can be scored in the random-split table and in the time-split table.
"""
import json

import numpy as np

from src import config
from src.prepare import dataset

SUBSET_DIR = config.RESULTS_DIR / "subsets"


def subset_path():
    return SUBSET_DIR / "groq_subset.json"


def groq_subset_ids(size=config.GROQ_SUBSET_SIZE, processed_dir=config.PROCESSED_DIR):
    """About 1000 ids that are test items in both splits, chosen with a fixed seed.

    Only the ids are looked at here, never the labels. The list is saved so it is visible in the repo.
    """
    path = subset_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    frame = dataset.load_all(processed_dir)
    in_both = (frame["split_random"] == "test") & (frame["split_time"] == "test")
    ids = sorted(frame.loc[in_both, "odi_no"])
    rng = np.random.default_rng(config.SEED)
    chosen = sorted(rng.choice(ids, size=min(size, len(ids)), replace=False).tolist())

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen), encoding="utf-8")
    return chosen
