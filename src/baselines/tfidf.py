"""Baseline 1: TF-IDF features and logistic regression.

TF-IDF turns a text into numbers: every word (and word pair) gets a weight that is high when the
word is frequent in this complaint but rare in the other complaints. Logistic regression then learns
which weights point to which component. It is a strong, fast, classical baseline.
"""
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src import config

C_GRID = [0.3, 1, 3, 10, 30]


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True, lowercase=True)


def make_classifier(c):
    return LogisticRegression(C=c, max_iter=300, random_state=config.SEED)


def tune(train, val):
    """Try several values of C on the validation set. Returns the best C and the full table."""
    vectorizer = make_vectorizer()
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(val["text"])

    rows = []
    for c in C_GRID:
        model = make_classifier(c).fit(x_train, train["label"])
        predicted = model.predict(x_val)
        rows.append({"C": c, "val_macro_f1": float(f1_score(val["label"], predicted, average="macro", zero_division=0)),
                     "val_accuracy": float((predicted == val["label"]).mean())})
    best = max(rows, key=lambda row: row["val_macro_f1"])
    return best["C"], rows


def fit(train, c):
    """Train on the training set. Returns (vectorizer, classifier)."""
    vectorizer = make_vectorizer()
    classifier = make_classifier(c).fit(vectorizer.fit_transform(train["text"]), train["label"])
    return vectorizer, classifier


def predict(model, texts):
    """Return the predicted labels and the confidence (the highest class probability) of each text."""
    vectorizer, classifier = model
    probabilities = classifier.predict_proba(vectorizer.transform(texts))
    best = probabilities.argmax(axis=1)
    return classifier.classes_[best], probabilities.max(axis=1)


def measure_latency(model, texts, n=200):
    """Median time in milliseconds to classify ONE text (one call per text)."""
    times = []
    for text in texts[:n]:
        started = time.perf_counter()
        predict(model, [text])
        times.append((time.perf_counter() - started) * 1000)
    return float(np.median(times))
