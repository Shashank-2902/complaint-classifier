"""Create the small test file tests/fixtures/raw_sample.tsv from the real NHTSA data.

It has the same 51-field layout as the raw flat file, but only the fields we use are filled.
It contains single-component complaints of every class, multi-component complaints, exact and
near duplicates, non-vehicle products and empty texts, so every preparation step has something to do.
Run once, only needed when the fixture has to be rebuilt:  python scripts/make_test_fixture.py
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.prepare import clean, labels, load_raw  # noqa: E402

PRIVATE = re.compile(r"@|\d{3}[-. ]\d{3,4}[-. ]\d{4}|\b[A-HJ-NPR-Z0-9]{17}\b|\bhttps?:|www\.", re.IGNORECASE)
rng = random.Random(config.SEED)


def row(cmpl_id, odi_no, component, received, text, source="IVOQ", product="V"):
    fields = [""] * 51
    for position, value in {0: cmpl_id, 1: odi_no, 11: component, 16: received, 19: text, 20: source, 45: product}.items():
        fields[position] = value
    return "\t".join(fields)


def main():
    raw = load_raw.read_raw_zip(config.RAW_DIR / "COMPLAINTS_RECEIVED_2020-2024.zip")
    raw = pd.concat([raw, load_raw.read_raw_zip(config.RAW_DIR / "COMPLAINTS_RECEIVED_2015-2019.zip")])
    raw["text"] = raw["text"].map(clean.clean_text)
    raw["top"] = raw["component"].map(labels.top_level_component)
    ok = raw[(raw["text"].str.len().between(80, 700)) & ~raw["text"].str.contains(PRIVATE) & raw["received"].notna()]
    vehicles = ok[ok["product_type"] == "V"]
    counts = vehicles.groupby("odi_no")["top"].nunique()
    singles = vehicles[vehicles["odi_no"].map(counts) == 1].drop_duplicates("odi_no")

    classes = ["ENGINE", "AIR BAGS", "POWER TRAIN", "ELECTRICAL SYSTEM", "STEERING", "SERVICE BRAKES", "STRUCTURE",
               "FUEL/PROPULSION SYSTEM", "EXTERIOR LIGHTING", "VISIBILITY/WIPER", "SUSPENSION",
               "ENGINE AND ENGINE COOLING", "VEHICLE SPEED CONTROL", "FORWARD COLLISION AVOIDANCE"]
    lines = []
    for name in classes:
        pool = singles[singles["top"] == name]
        lines += [(r.cmpl_id, r.odi_no, r.component, r.received, r.text, r.source, "V")
                  for r in pool.sample(n=min(24, len(pool)), random_state=config.SEED).itertuples()]
    other = singles[~singles["top"].isin(classes)]
    lines += [(r.cmpl_id, r.odi_no, r.component, r.received, r.text, r.source, "V")
              for r in other.sample(n=60, random_state=config.SEED).itertuples()]

    multi_ids = counts[counts > 1].index
    multi = vehicles[vehicles["odi_no"].isin(multi_ids)]
    multi = multi[multi["odi_no"].isin(pd.Series(multi["odi_no"].unique()).sample(n=12, random_state=config.SEED))]
    lines += [(r.cmpl_id, r.odi_no, r.component, r.received, r.text, r.source, "V") for r in multi.itertuples()]

    base = list(lines[:200:20])           # exact and near copies of existing complaints, with new ids
    for i, (cid, odi, comp, rec, text, src, prod) in enumerate(base):
        lines.append((f"9{i:08d}", f"9{i:08d}", comp, rec, text, src, prod))                          # exact duplicate
    for i, (cid, odi, comp, rec, text, src, prod) in enumerate(base[:5]):
        lines.append((f"8{i:08d}", f"8{i:08d}", comp, rec, text + " PLEASE HELP", src, prod))         # near duplicate

    for i, (kind, comp) in enumerate([("T", "TIRES"), ("T", "TIRES"), ("E", "EQUIPMENT"), ("C", "CHILD SEAT")]):
        lines.append((f"7{i:08d}", f"7{i:08d}", comp, "20220101", "The product failed while in normal use on the road.", "IVOQ", kind))
    for i, text in enumerate(["", "too short"]):
        lines.append((f"6{i:08d}", f"6{i:08d}", "ENGINE", "20220101", text, "IVOQ", "V"))

    rng.shuffle(lines)
    path = ROOT / "tests" / "fixtures" / "raw_sample.tsv"
    path.write_text("\n".join(row(*line) for line in lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} rows to {path}")


if __name__ == "__main__":
    main()
