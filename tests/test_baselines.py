import numpy as np
import pytest

from src.baselines import embeddings, run_baseline, tfidf
from src.prepare import dataset


@pytest.fixture
def small_split(prepared_dir):
    return dataset.load_split("random", prepared_dir)


def test_tfidf_tuning_picks_a_c_from_the_grid(small_split):
    best_c, grid = tfidf.tune(small_split["train"], small_split["val"])
    assert best_c in tfidf.C_GRID
    assert len(grid) == len(tfidf.C_GRID)


def test_tfidf_predictions_have_labels_and_confidences(small_split):
    model = tfidf.fit(small_split["train"], c=3)
    labels, confidence = tfidf.predict(model, small_split["val"]["text"])
    assert len(labels) == len(small_split["val"])
    assert set(labels) <= set(small_split["train"]["label"])
    assert ((confidence > 0) & (confidence <= 1)).all()


def test_tfidf_beats_guessing_the_most_common_class(small_split):
    model = tfidf.fit(small_split["train"], c=10)
    labels, _ = tfidf.predict(model, small_split["val"]["text"])
    accuracy = (labels == small_split["val"]["label"]).mean()
    most_common = small_split["train"]["label"].value_counts(normalize=True).iloc[0]
    assert accuracy > most_common


def test_tuning_code_cannot_reach_the_test_set(monkeypatch, small_split, tmp_path):
    """The tune stage must only call load_split. If it opened the test set, this test fails."""
    def forbidden(*args, **kwargs):
        raise AssertionError("the tune stage opened the test set")

    monkeypatch.setattr(run_baseline.dataset, "load_test_once", forbidden)
    monkeypatch.setattr(run_baseline.dataset, "load_split", lambda scheme: small_split)
    monkeypatch.setattr(run_baseline.dataset, "class_list", lambda: sorted(set(small_split["train"]["label"])))
    monkeypatch.setattr(run_baseline.config, "TUNING_DIR", tmp_path)
    result = run_baseline.tune("tfidf_lr", "random")
    assert result["best_C"] in tfidf.C_GRID
    assert len(result["confidence_thresholds"]) == 2


class FakeEncoder:
    """Stands in for the sentence-transformer, so the tests need no model download."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.array([[len(t) % 7, len(t.split()) % 5, hash(t) % 3 + 1.0] for t in texts], dtype=float)


def test_embeddings_are_cached_and_reused(small_split, tmp_path):
    frame = small_split["train"]
    cache = tmp_path / "emb.npz"
    encoder = FakeEncoder()
    first = embeddings.embed_dataset(frame, encoder, cache)
    calls_after_first = encoder.calls
    second = embeddings.embed_dataset(frame, FakeEncoder(), cache)          # a new encoder must not be needed
    assert encoder.calls == calls_after_first
    assert np.allclose(first, second)


def test_changed_data_invalidates_the_embedding_cache(small_split, tmp_path):
    cache = tmp_path / "emb.npz"
    embeddings.embed_dataset(small_split["train"], FakeEncoder(), cache)
    encoder = FakeEncoder()
    embeddings.embed_dataset(small_split["val"], encoder, cache)           # different rows, so the cache is not valid
    assert encoder.calls > 0


def test_embedding_classifier_tunes_and_predicts(small_split):
    rng = np.random.default_rng(0)
    classes = sorted(set(small_split["train"]["label"]))
    centers = {name: rng.normal(size=8) for name in classes}
    make = lambda frame: np.array([centers[label] + rng.normal(scale=0.3, size=8) for label in frame["label"]])
    x_train, x_val = make(small_split["train"]), make(small_split["val"])
    best_c, grid = embeddings.tune(x_train, small_split["train"]["label"], x_val, small_split["val"]["label"])
    assert best_c in embeddings.C_GRID
    classifier = embeddings.make_classifier(best_c).fit(x_train, small_split["train"]["label"])
    labels, confidence = embeddings.predict(classifier, x_val)
    assert (labels == small_split["val"]["label"]).mean() > 0.8
    assert confidence.max() <= 1
