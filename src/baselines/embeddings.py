"""Baseline 2: sentence embeddings and logistic regression.

A sentence-transformer reads a whole text and returns a vector of numbers (384 here) that captures
its meaning. Texts with similar meaning get similar vectors, even if they use different words.
Logistic regression is then trained on these vectors. This needs no GPU.
"""
import hashlib
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src import config

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
C_GRID = [1, 3, 10, 30, 100]
CACHE_FILE = config.PROCESSED_DIR / "embeddings_minilm.npz"


def load_encoder():
    from sentence_transformers import SentenceTransformer   # imported here: only needed for real runs

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(encoder, texts, batch_size=64):
    """Return one normalised vector per text."""
    return encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def ids_fingerprint(ids):
    return hashlib.sha256(",".join(ids).encode()).hexdigest()[:16]


def embed_dataset(frame, encoder=None, cache_file=CACHE_FILE):
    """Embed every complaint once and keep the result on disk. Returns an array in the order of frame.

    The encoder is only loaded if the cache file is missing or does not match the data.
    """
    ids = frame["odi_no"].tolist()
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=False)
        if str(cached["fingerprint"]) == ids_fingerprint(ids):
            return cached["vectors"]

    encoder = encoder or load_encoder()
    parts = []
    for start in range(0, len(frame), 2000):
        parts.append(embed(encoder, frame["text"].iloc[start:start + 2000]))
        print(f"  embedded {min(start + 2000, len(frame))} of {len(frame)}", flush=True)
    vectors = np.vstack(parts)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, vectors=vectors, fingerprint=ids_fingerprint(ids))
    return vectors


def make_classifier(c):
    return LogisticRegression(C=c, max_iter=500, random_state=config.SEED)


def tune(x_train, y_train, x_val, y_val):
    rows = []
    for c in C_GRID:
        predicted = make_classifier(c).fit(x_train, y_train).predict(x_val)
        rows.append({"C": c, "val_macro_f1": float(f1_score(y_val, predicted, average="macro", zero_division=0)),
                     "val_accuracy": float((predicted == y_val).mean())})
    best = max(rows, key=lambda row: row["val_macro_f1"])
    return best["C"], rows


def predict(classifier, vectors):
    probabilities = classifier.predict_proba(vectors)
    return classifier.classes_[probabilities.argmax(axis=1)], probabilities.max(axis=1)


def measure_latency(encoder, classifier, texts, n=200):
    """Median time in milliseconds to classify ONE text, including the embedding step."""
    times = []
    for text in texts[:n]:
        started = time.perf_counter()
        predict(classifier, embed(encoder, [text]))
        times.append((time.perf_counter() - started) * 1000)
    return float(np.median(times))
