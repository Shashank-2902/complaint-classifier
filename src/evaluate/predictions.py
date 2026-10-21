"""Every system writes its predictions in the same simple format, so one evaluation can read them all.

results/predictions/<system>__<scheme>.csv        odi_no, pred, confidence
results/predictions/<system>__<scheme>.meta.json  latency, cost, model, notes
"""
import json

import pandas as pd

from src import config


def paths(system, scheme, directory=config.PREDICTIONS_DIR):
    stem = f"{system}__{scheme}"
    return directory / f"{stem}.csv", directory / f"{stem}.meta.json"


def write_predictions(system, scheme, ids, preds, confidences, meta, directory=config.PREDICTIONS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = paths(system, scheme, directory)
    pd.DataFrame({"odi_no": ids, "pred": preds, "confidence": confidences}).to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_predictions(system, scheme, directory=config.PREDICTIONS_DIR):
    """Return (predictions frame, meta dict), or (None, None) if this system was not run."""
    csv_path, meta_path = paths(system, scheme, directory)
    if not csv_path.exists():
        return None, None
    frame = pd.read_csv(csv_path, dtype={"odi_no": str})
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return frame, meta
