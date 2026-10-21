import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.evaluate import metrics, predictions, report

CLASSES = ["A", "B", "C"]


def test_metrics_match_scikit_learn():
    true = ["A", "A", "B", "B", "C", "A"]
    pred = ["A", "B", "B", "B", "A", "A"]
    result = metrics.compute_metrics(true, pred, CLASSES)
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(f1_score(true, pred, labels=CLASSES, average="macro", zero_division=0))
    assert result["per_class_f1"]["C"] == 0.0           # class C was never predicted correctly


def test_macro_f1_is_lower_than_accuracy_when_a_small_class_is_missed():
    true = ["A"] * 90 + ["B"] * 10
    pred = ["A"] * 100
    result = metrics.compute_metrics(true, pred, ["A", "B"])
    assert result["accuracy"] == 0.9
    assert result["macro_f1"] < 0.5


def test_confusion_table_counts_and_shape():
    table = metrics.confusion_table(["A", "A", "B"], ["A", "B", "B"], CLASSES)
    assert table.loc["A", "A"] == 1 and table.loc["A", "B"] == 1 and table.loc["B", "B"] == 1
    assert list(table.columns) == CLASSES + ["INVALID"]


def test_top_confusions_are_sorted_by_count():
    true = ["A", "A", "A", "B", "B", "C"]
    pred = ["B", "B", "B", "A", "C", "C"]
    assert metrics.top_confusions(true, pred, k=2) == [("A", "B", 3), ("B", "A", 1)]


def test_thresholds_are_chosen_on_validation_confidences():
    confidence = np.linspace(0.1, 1.0, 100)
    high, low = metrics.choose_thresholds(confidence, coverages=(0.7, 0.4))
    assert low > high
    assert (confidence >= high).mean() == pytest.approx(0.7, abs=0.02)
    assert (confidence >= low).mean() == pytest.approx(0.4, abs=0.02)


def test_accuracy_and_coverage_at_a_threshold():
    true = np.array(["A", "A", "B", "B"])
    pred = np.array(["A", "B", "B", "A"])
    confidence = np.array([0.9, 0.3, 0.8, 0.4])
    accuracy, coverage = metrics.accuracy_at_threshold(true, pred, confidence, 0.5)
    assert accuracy == 1.0 and coverage == 0.5
    accuracy, coverage = metrics.accuracy_at_threshold(true, pred, confidence, 0.99)
    assert np.isnan(accuracy) and coverage == 0.0


# ---------- predictions files and the results table ----------

def test_predictions_roundtrip(tmp_path):
    predictions.write_predictions("tfidf_lr", "time", ["1", "2"], ["A", "B"], [0.9, 0.4], {"latency_ms_per_item": 1.5}, tmp_path)
    frame, meta = predictions.read_predictions("tfidf_lr", "time", tmp_path)
    assert frame["pred"].tolist() == ["A", "B"] and meta["latency_ms_per_item"] == 1.5


def test_a_system_that_was_not_run_is_reported_as_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(predictions.config, "PREDICTIONS_DIR", tmp_path)
    frame, meta = predictions.read_predictions("groq_zero_shot", "time", tmp_path)
    assert frame is None and meta is None
    assert "not run" in report.format_row("Groq zero-shot", None)


def test_the_table_never_invents_numbers(tmp_path, monkeypatch):
    """With no prediction files at all, every system in the table says 'not run'."""
    monkeypatch.setattr("src.evaluate.report.read_predictions", lambda system, scheme: (None, None))
    truth = pd.DataFrame({"odi_no": ["1"], "label": ["A"]})
    table, results = report.results_table("time", truth, CLASSES)
    assert all(result is None for result in results.values())
    assert table.count("not run") == 5 * 4


def test_system_result_uses_only_items_that_were_predicted(tmp_path, monkeypatch):
    predictions.write_predictions("groq_zero_shot", "time", ["1", "2"], ["A", "INVALID"], [None, None],
                                  {"latency_ms_per_item": 300, "cost_per_1000_usd": 0.02}, tmp_path)
    monkeypatch.setattr("src.evaluate.report.read_predictions", lambda s, scheme: predictions.read_predictions(s, scheme, tmp_path))
    truth = pd.DataFrame({"odi_no": ["1", "2", "3"], "label": ["A", "B", "C"]})
    result = report.system_result("groq_zero_shot", "time", truth, CLASSES)
    assert result["n"] == 2 and result["accuracy"] == 0.5
    assert result["cost_per_1000"] == 0.02
