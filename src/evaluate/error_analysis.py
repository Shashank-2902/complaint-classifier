"""Confusion matrices and real misclassified examples.

    python -m src.evaluate.error_analysis --system tfidf_lr --scheme time

Writes results/confusion_<system>__<scheme>.csv and .png, and results/errors_<system>__<scheme>.json
with the five most frequent mistakes and a few real texts for each. Reading the examples and judging
whether the model or the label was wrong is done by hand, see results/error_analysis.md.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.evaluate.predictions import read_predictions  # noqa: E402
from src.prepare import dataset  # noqa: E402


def draw_confusion(table, title, path):
    shares = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(shares.values, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(table.index)), table.index, fontsize=7)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(title)
    figure.colorbar(image, label="share of the true class")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def run(system, scheme, examples=3):
    predictions, _ = read_predictions(system, scheme)
    if predictions is None:
        raise SystemExit(f"{system} / {scheme} was not run.")
    classes = dataset.class_list()
    truth = dataset.load_all()
    merged = predictions.merge(truth[["odi_no", "label", "text", "received"]], on="odi_no")

    table = metrics.confusion_table(merged["label"], merged["pred"], classes)
    stem = f"{system}__{scheme}"
    table.to_csv(config.RESULTS_DIR / f"confusion_{stem}.csv")
    draw_confusion(table, f"{config.SYSTEM_NAMES[system]} ({scheme} split)", config.RESULTS_DIR / f"confusion_{stem}.png")

    report = []
    for true_label, pred_label, count in metrics.top_confusions(merged["label"], merged["pred"], k=5):
        wrong = merged[(merged["label"] == true_label) & (merged["pred"] == pred_label)]
        sample = wrong.sample(n=min(examples, len(wrong)), random_state=config.SEED)
        report.append({"true": true_label, "predicted": pred_label, "count": count,
                       "examples": [{"odi_no": r.odi_no, "received": r.received, "text": r.text[:600]} for r in sample.itertuples()]})
    (config.RESULTS_DIR / f"errors_{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entry in report:
        print(f"{entry['count']:5}  true {entry['true']}  ->  predicted {entry['predicted']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=config.SYSTEMS)
    parser.add_argument("--scheme", required=True, choices=config.SPLIT_SCHEMES)
    args = parser.parse_args()
    run(args.system, args.scheme)


if __name__ == "__main__":
    main()
