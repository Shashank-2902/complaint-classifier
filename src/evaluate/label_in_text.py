"""Does the complaint text already contain its own label name?

NHTSA staff often write the recalled component in brackets in the narrative, for example
"... NHTSA CAMPAIGN NUMBER: 14V355000 (ELECTRICAL SYSTEM) ...". If the label name is in the text, a
word-based model like TF-IDF can read the label directly, so such complaints are easier.

Matching rule (the main one):
    the complaint text contains "(" + the complaint's own label name + ")", ignoring case.
    For the class OTHER, the NHTSA name "UNKNOWN OR OTHER" also counts.
It is a plain text match with no fuzzy matching. Brackets are required because the bare words
("engine", "steering", "other") appear in many complaints for ordinary reasons.

    python -m src.evaluate.label_in_text

This only reads local files: data/processed and results/predictions. It does not open the test set
through the guarded function (that would be logged), it reads the labels the same way
src/evaluate/report.py does and keeps the rows that the TF-IDF predictions cover.
"""
import re

import pandas as pd

from src import config
from src.evaluate.predictions import read_predictions
from src.prepare import dataset

SYSTEM = "tfidf_lr"
OUTPUT = config.RESULTS_DIR / "label_in_text.md"
EXTRA_NAMES = {config.OTHER: ["UNKNOWN OR OTHER"]}


def names_for(label):
    return [label] + EXTRA_NAMES.get(label, [])


def bracket_match(text, label):
    """True if the text contains (LABEL), ignoring case."""
    return any(re.search(r"\(" + re.escape(name) + r"\)", text, flags=re.IGNORECASE) for name in names_for(label))


def plain_match(text, label):
    """The looser alternative: the label words anywhere in the text. Reported only for comparison."""
    return any(re.search(re.escape(name), text, flags=re.IGNORECASE) for name in names_for(label))


def accuracy(frame):
    return float((frame["pred"] == frame["label"]).mean()) if len(frame) else float("nan")


def summarise(frame, matcher):
    """frame has the columns text, label and pred. Returns the counts and the accuracy on both groups."""
    found = frame.apply(lambda row: matcher(row["text"], row["label"]), axis=1)
    return {
        "n": len(frame),
        "with_label": int(found.sum()),
        "share": float(found.mean()) if len(frame) else float("nan"),
        "accuracy_with": accuracy(frame[found]),
        "accuracy_without": accuracy(frame[~found]),
    }


def test_rows(scheme):
    """The test complaints that the TF-IDF predictions cover, with label, text and prediction."""
    predictions, _ = read_predictions(SYSTEM, scheme)
    if predictions is None:
        raise SystemExit(f"no predictions for {SYSTEM} on the {scheme} split")
    truth = dataset.load_all()
    return predictions.merge(truth[["odi_no", "label", "text"]], on="odi_no", how="inner")


def block(scheme, main, loose):
    pct = lambda value: f"{100 * value:.1f} percent"  # noqa: E731
    return [
        f"## {scheme.capitalize()} split ({main['n']:,} test complaints)", "",
        "| Rule | Complaints with the label name | Share | TF-IDF accuracy on them | TF-IDF accuracy on the rest |",
        "| --- | --- | --- | --- | --- |",
        f"| (LABEL) in brackets (main rule) | {main['with_label']:,} | {pct(main['share'])} | {main['accuracy_with']:.3f} | {main['accuracy_without']:.3f} |",
        f"| label words anywhere (looser) | {loose['with_label']:,} | {pct(loose['share'])} | {loose['accuracy_with']:.3f} | {loose['accuracy_without']:.3f} |",
        "",
    ]


def main():
    lines = ["# Does the complaint text contain its own label name?", "",
             "Matching rule (main): the text contains `(` + the complaint's own label name + `)`, ignoring case, for example "
             "`(ENGINE AND ENGINE COOLING)`. For the class OTHER, the NHTSA name `UNKNOWN OR OTHER` also counts. No fuzzy matching.", "",
             "Why brackets: NHTSA staff write the recalled component in brackets after a campaign number, for example "
             "`NHTSA CAMPAIGN NUMBER: 14V355000 (ELECTRICAL SYSTEM)`. The bare words (`engine`, `steering`, `other`) appear in many complaints "
             "for ordinary reasons, so the looser rule in the second row is shown only for comparison.", "",
             "What is ambiguous: the bracket text is the component of the recall, which is not always the complaint's label, and a complaint can "
             "name a different component in brackets. This file counts only a match with the complaint's own label. A match does not prove that "
             "the model used it. The accuracy difference also mixes in that these complaints are mostly about recalls, which may be easier or harder for other reasons.", "",
             f"Predictions: `{SYSTEM}` on the complete test sets, from `results/predictions`.", ""]
    for scheme in config.SPLIT_SCHEMES:
        frame = test_rows(scheme)
        lines += block(scheme, summarise(frame, bracket_match), summarise(frame, plain_match))
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
