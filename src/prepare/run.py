"""Build the dataset from the raw NHTSA files.

    python -m src.prepare.run

Steps, and the numbers of each step are saved in results/data_report.json:
  1. read the raw files                          5. remove exact duplicate texts
  2. keep vehicle complaints                     6. map labels (top classes + OTHER)
  3. clean the text, drop very short texts       7. random sample, then remove near duplicates
  4. drop multi-component complaints             8. create the random and the time split
"""
import argparse
import hashlib
import json

import pandas as pd

from src import config
from src.prepare import clean, dedupe, labels, load_raw, splits
from src.prepare.dataset import DATASET_FILE


def fingerprint(ids):
    """A short code for a list of ids. Used to check that two machines made the same splits."""
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]


def build(raw, sample_size=config.SAMPLE_SIZE, top_k=config.TOP_K, seed=config.SEED):
    """Run the whole preparation on a raw frame. Returns (dataset, report)."""
    report = {"raw_rows": len(raw)}

    frame = raw[raw["product_type"] == "V"].copy()
    report["vehicle_rows"] = len(frame)

    frame["received"] = pd.to_datetime(frame["received"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["received", "odi_no"])
    frame, report["too_short_or_empty_text"] = clean.clean_frame(frame)

    frame["top_level"] = frame["component"].map(labels.top_level_component)
    frame = frame[frame["top_level"] != ""]
    report["complaints_before_multi_filter"] = int(frame["odi_no"].nunique())
    frame, report["multi_component_complaints_dropped"] = labels.drop_multi_component(frame)
    report["single_component_complaints"] = len(frame)

    frame, report["exact_duplicates_removed"] = dedupe.drop_exact_duplicates(frame)
    report["after_exact_dedupe"] = len(frame)

    classes = labels.most_frequent_classes(frame, top_k)
    frame = labels.map_labels(frame, classes)
    report["classes"] = classes

    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    report["sampled"] = len(frame)
    frame, report["near_duplicates_removed"] = dedupe.drop_near_duplicates(frame)
    report["final_complaints"] = len(frame)

    frame = splits.add_splits(frame.reset_index(drop=True))
    report.update(describe(frame))
    return frame, report


def describe(frame):
    """Counts and date ranges of the final data (no complaint text)."""
    info = {"label_counts": frame["label"].value_counts().to_dict(), "splits": {}}
    for scheme in config.SPLIT_SCHEMES:
        column = f"split_{scheme}"
        info["splits"][scheme] = {}
        for part in ["train", "val", "test"]:
            rows = frame[frame[column] == part]
            info["splits"][scheme][part] = {
                "rows": len(rows),
                "first_date": str(rows["received"].min().date()),
                "last_date": str(rows["received"].max().date()),
                "fingerprint": fingerprint(rows["odi_no"].tolist()),
            }
    return info


def save(frame, report, processed_dir=config.PROCESSED_DIR, results_dir=config.RESULTS_DIR):
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    keep = ["odi_no", "received", "source", "top_level", "label", "text", "split_random", "split_time"]
    frame[keep].to_csv(processed_dir / DATASET_FILE, index=False)
    (results_dir / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build the dataset from the raw NHTSA files.")
    parser.add_argument("--sample-size", type=int, default=config.SAMPLE_SIZE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    print("Reading raw files ...")
    raw = load_raw.read_raw_dir()
    frame, report = build(raw, args.sample_size, args.top_k)
    save(frame, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("splits", "label_counts")}, indent=2))


if __name__ == "__main__":
    main()
