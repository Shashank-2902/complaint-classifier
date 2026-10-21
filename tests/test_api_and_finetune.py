import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main
from src.baselines import tfidf
from src.finetune import data
from src.prepare import dataset


@pytest.fixture
def client(prepared_dir, tmp_path, monkeypatch):
    """The API with a small model trained on the fixture data."""
    frames = dataset.load_split("random", prepared_dir)
    vectorizer, classifier = tfidf.fit(frames["train"], c=10)
    bundle = {"vectorizer": vectorizer, "classifier": classifier, "classes": list(classifier.classes_), "thresholds": [0.4, 0.6]}
    monkeypatch.setitem(main.state, "bundle", bundle)
    return TestClient(main.app)


TEXT = "The air bag warning light stayed on and the air bag did not deploy during the crash at highway speed."


def test_classify_returns_label_confidence_and_top_three(client):
    answer = client.post("/classify", json={"text": TEXT}).json()
    assert set(answer) == {"label", "confidence", "needs_human_review", "threshold_used", "top"}
    assert len(answer["top"]) == 3 and answer["top"][0]["label"] == answer["label"]
    assert 0 < answer["confidence"] <= 1


def test_low_confidence_goes_to_human_review(client):
    strict = client.post("/classify", json={"text": TEXT, "threshold": 1.0}).json()
    lenient = client.post("/classify", json={"text": TEXT, "threshold": 0.0}).json()
    assert strict["needs_human_review"] is True and strict["threshold_used"] == 1.0
    assert lenient["needs_human_review"] is False


def test_default_threshold_comes_from_the_model(client):
    assert client.post("/classify", json={"text": TEXT}).json()["threshold_used"] == 0.4


def test_bad_requests_are_refused(client):
    assert client.post("/classify", json={"text": "short"}).status_code == 422
    assert client.post("/classify", json={"text": TEXT, "threshold": 2}).status_code == 422


def test_missing_model_gives_503_with_instructions(monkeypatch, tmp_path):
    monkeypatch.setitem(main.state, "bundle", None)
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "nothing.joblib")
    answer = TestClient(main.app).post("/classify", json={"text": TEXT})
    assert answer.status_code == 503 and "export_model" in answer.json()["detail"]


# ---------- fine-tuning helpers (no GPU, no torch) ----------

def test_label_maps_are_inverse_of_each_other():
    to_id, to_label = data.label_maps(["A", "B", "OTHER"])
    assert to_id == {"A": 0, "B": 1, "OTHER": 2} and to_label[2] == "OTHER"
    assert data.encode_labels(["B", "A"], to_id).tolist() == [1, 0]


def test_batches_cover_every_item_once():
    batches = data.make_batches(10, 4, seed=1)
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert [b.tolist() for b in data.make_batches(10, 4, seed=1)] == [b.tolist() for b in batches]   # same seed, same order


def test_softmax_rows_sum_to_one_and_predictions_use_the_highest_probability():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 5.0, 1.0]])
    assert np.allclose(data.softmax(logits).sum(axis=1), 1)
    labels, confidence = data.to_predictions(logits, {0: "A", 1: "B", 2: "C"})
    assert labels == ["A", "B"] and confidence[1] > confidence[0]
