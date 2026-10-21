"""Build the results table from the prediction files.

    python -m src.evaluate.report

It only reads files. A system that has no prediction file is shown as "not run".
"""
import json

import pandas as pd

from src import config
from src.evaluate import metrics, subsets
from src.evaluate.predictions import read_predictions
from src.prepare import dataset

NOT_RUN = "not run"


def system_result(system, scheme, truth, classes, restrict_to=None):
    """Metrics of one system on one split. restrict_to limits the items (used for the common subset)."""
    predictions, meta = read_predictions(system, scheme)
    if predictions is None:
        return None
    merged = predictions.merge(truth[["odi_no", "label"]], on="odi_no", how="inner")
    if restrict_to is not None:
        merged = merged[merged["odi_no"].isin(restrict_to)]
    if merged.empty:
        return None
    result = metrics.compute_metrics(merged["label"], merged["pred"], classes)
    result.update(latency_ms=meta.get("latency_ms_per_item"), cost_per_1000=meta.get("cost_per_1000_usd"),
                  latency_device=meta.get("latency_device"), invalid=meta.get("invalid_answers"), api_model=meta.get("provider"))
    return result


def format_row(name, result):
    if result is None:
        return f"| {name} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} |"
    if result["cost_per_1000"]:
        cost = f"${result['cost_per_1000']:.3f}"
    else:
        cost = "$0 (free plan)" if result.get("api_model") else "0 (local)"
    return (f"| {name} | {result['n']} | {result['accuracy']:.3f} | {result['macro_f1']:.3f} | "
            f"{result['latency_ms']:.1f} | {cost} |")


def results_table(scheme, truth, classes, restrict_to=None):
    lines = ["| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |",
             "| --- | --- | --- | --- | --- | --- |"]
    results = {}
    for system in config.SYSTEMS:
        results[system] = system_result(system, scheme, truth, classes, restrict_to)
        lines.append(format_row(config.SYSTEM_NAMES[system], results[system]))
    return "\n".join(lines), results


def per_class_table(results, classes):
    shown = {s: r for s, r in results.items() if r is not None}
    if not shown:
        return "No system was run on this split."
    header = "| Class | " + " | ".join(config.SYSTEM_NAMES[s] for s in shown) + " |"
    lines = [header, "| --- | " + " | ".join("---" for _ in shown) + " |"]
    for name in classes:
        lines.append(f"| {name} | " + " | ".join(f"{r['per_class_f1'][name]:.2f}" for r in shown.values()) + " |")
    return "\n".join(lines)


def threshold_table(scheme, truth):
    """Accuracy and coverage at two confidence thresholds, chosen on the validation set."""
    lines = ["| System | Threshold | Accuracy of accepted items | Coverage (share accepted) |", "| --- | --- | --- | --- |"]
    shown = False
    for system in config.SYSTEMS:
        predictions, _ = read_predictions(system, scheme)
        tuning_file = config.TUNING_DIR / f"{system}__{scheme}.json"
        if predictions is None or predictions["confidence"].isna().all() or not tuning_file.exists():
            continue
        thresholds = json.loads(tuning_file.read_text(encoding="utf-8"))["confidence_thresholds"]
        merged = predictions.merge(truth[["odi_no", "label"]], on="odi_no")
        for threshold in thresholds:
            accuracy, coverage = metrics.accuracy_at_threshold(merged["label"], merged["pred"], merged["confidence"], threshold)
            lines.append(f"| {config.SYSTEM_NAMES[system]} | {threshold:.3f} | {accuracy:.3f} | {coverage:.3f} |")
            shown = True
    return "\n".join(lines) if shown else "Not available: no system with confidence scores and a tuning file was run."


def split_summary(report):
    lines = ["| Part | Rows | First date | Last date |", "| --- | --- | --- | --- |"]
    out = {}
    for scheme, parts in report["splits"].items():
        out[scheme] = "\n".join(lines + [f"| {part} | {info['rows']} | {info['first_date']} | {info['last_date']} |"
                                        for part, info in parts.items()])
    return out


def build_markdown():
    truth = dataset.load_all()
    classes = dataset.class_list()
    report = json.loads((config.RESULTS_DIR / "data_report.json").read_text(encoding="utf-8"))
    summaries = split_summary(report)

    parts = ["# Results", "",
             "Accuracy is measured against the NHTSA component label, which was assigned by NHTSA staff or by the complaint "
             "form. That label has noise, so these numbers are agreement with the label, not with absolute truth.", ""]
    for scheme in config.SPLIT_SCHEMES:
        full_table, full_results = results_table(scheme, truth, classes)
        subset_ids = set(subsets.groq_subset_ids()) if subsets.subset_path().exists() else None
        parts += [f"## {scheme.capitalize()} split", "", summaries[scheme], "", "### All test items", "", full_table, ""]
        if subset_ids:
            subset_table, _ = results_table(scheme, truth, classes, restrict_to=subset_ids)
            parts += [f"### The fixed subset of {len(subset_ids)} test items (the same items for every system)", "", subset_table, ""]
        parts += ["### F1 per class (all test items)", "", per_class_table(full_results, classes), "",
                  "### Confidence threshold (items below the threshold would go to a human)", "", threshold_table(scheme, truth), ""]
    return "\n".join(parts)


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = build_markdown()
    (config.RESULTS_DIR / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
