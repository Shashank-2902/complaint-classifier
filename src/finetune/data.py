"""Helpers for fine-tuning that need no GPU and no torch, so they can be tested anywhere."""
import numpy as np


def label_maps(classes):
    """Return (label -> number, number -> label). Models work with numbers, not names."""
    to_id = {name: i for i, name in enumerate(classes)}
    to_label = {i: name for name, i in to_id.items()}
    return to_id, to_label


def encode_labels(labels, to_id):
    return np.array([to_id[label] for label in labels], dtype=np.int64)


def make_batches(n_items, batch_size, seed=None):
    """Split the positions 0..n_items-1 into batches. With a seed they are shuffled first."""
    positions = np.arange(n_items)
    if seed is not None:
        np.random.default_rng(seed).shuffle(positions)
    return [positions[start:start + batch_size] for start in range(0, n_items, batch_size)]


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def to_predictions(logits, to_label):
    """Turn model outputs into (predicted label names, confidence = highest probability)."""
    probabilities = softmax(np.asarray(logits, dtype=np.float64))
    best = probabilities.argmax(axis=1)
    return [to_label[int(i)] for i in best], probabilities.max(axis=1)
