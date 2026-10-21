"""Metrics, confusion analysis and the confidence threshold."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(true, pred, classes):
    """Accuracy, macro-F1 and F1 per class. Macro-F1 counts every class equally."""
    per_class = f1_score(true, pred, labels=classes, average=None, zero_division=0)
    return {
        "n": len(true),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": {name: float(value) for name, value in zip(classes, per_class)},
    }


def confusion_table(true, pred, classes):
    """Rows are the true label, columns are the predicted label."""
    table = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="predicted"))
    return table.reindex(index=classes, columns=classes + ["INVALID"], fill_value=0)


def top_confusions(true, pred, k=5):
    """The k most frequent mistakes as (true label, predicted label, count)."""
    wrong = pd.DataFrame({"true": list(true), "pred": list(pred)})
    wrong = wrong[wrong["true"] != wrong["pred"]]
    counts = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False).head(k)
    return [(a, b, int(n)) for (a, b), n in counts.items()]


def choose_thresholds(val_confidence, coverages=(0.7, 0.4)):
    """Pick confidence thresholds on the validation set so that about 70 and 40 percent of items are accepted."""
    values = np.asarray(val_confidence, dtype=float)
    return [float(np.quantile(values, 1 - coverage)) for coverage in coverages]


def accuracy_at_threshold(true, pred, confidence, threshold):
    """Accuracy of the accepted items (confidence at or above the threshold) and the share that was accepted."""
    true, pred, confidence = np.asarray(true), np.asarray(pred), np.asarray(confidence, dtype=float)
    accepted = confidence >= threshold
    coverage = float(accepted.mean())
    accuracy = float((true[accepted] == pred[accepted]).mean()) if accepted.any() else float("nan")
    return accuracy, coverage
