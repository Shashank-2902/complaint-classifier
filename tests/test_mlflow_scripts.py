"""The MLflow scripts, run on the small fixture data with a temporary tracking folder. They need mlflow, not the data/ folder."""
import joblib
import pandas as pd
import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from scripts import export_registered_model, track_tfidf  # noqa: E402
from src.baselines import tfidf  # noqa: E402
from src.prepare import dataset  # noqa: E402

RESULTS_TEXT = """# Results

## Random split

### All test items

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 13840 | 0.767 | 0.742 | 2.7 | 0 (local) |
| Fine-tuned small encoder | 13840 | 0.803 | 0.790 | 5.6 | 0 (local) |

### The fixed subset

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 1000 | 0.775 | 0.745 | 2.7 | 0 (local) |

## Time split

### All test items

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 13840 | 0.713 | 0.674 | 2.2 | 0 (local) |
"""
TUNED = {"best_C": 10, "confidence_thresholds": [0.4, 0.6]}


def test_the_published_numbers_are_copied_from_the_right_split_and_table():
    random = track_tfidf.published_test_numbers(RESULTS_TEXT, "random")
    time = track_tfidf.published_test_numbers(RESULTS_TEXT, "time")
    assert random == {"published_test_scheme": "random", "published_test_items": "13840",
                      "published_test_accuracy": "0.767", "published_test_macro_f1": "0.742"}
    assert time["published_test_accuracy"] == "0.713" and time["published_test_macro_f1"] == "0.674"


def test_a_missing_row_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        track_tfidf.published_test_numbers(RESULTS_TEXT.replace("TF-IDF + logistic regression", "Something else"), "time")


def test_fingerprints_are_flattened_with_scheme_and_part():
    report = {"splits": {"time": {"train": {"fingerprint": "aaa"}, "val": {"fingerprint": "bbb"}}}}
    assert track_tfidf.data_fingerprints(report) == {"fingerprint_time_train": "aaa", "fingerprint_time_val": "bbb"}


@pytest.fixture
def tracked(built, prepared_dir, tmp_path):
    _, report = built
    result = track_tfidf.track("random", processed_dir=prepared_dir, tuned=TUNED, report=report, results_text=RESULTS_TEXT,
                               tracking_dir=tmp_path / "mlruns")
    return result, report, tmp_path


def test_the_run_has_parameters_validation_metrics_fingerprints_commit_and_published_tags(tracked):
    result, report, tmp_path = tracked
    track_tfidf.use_tracking_dir(tmp_path / "mlruns")
    run = MlflowClient().get_run(result["run_id"])
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    assert params["C"] == "10" and params["random_state"] == "42" and params["ngram_range"] == "1-2"
    assert set(metrics) == {"val_accuracy", "val_macro_f1"}            # no test metric is computed
    assert tags["published_test_accuracy"] == "0.767" and tags["source"] == "results/results.md"
    assert "git_commit" in tags
    fingerprints = {k: v for k, v in params.items() if k.startswith("fingerprint_")}
    assert fingerprints == track_tfidf.data_fingerprints(report)


def test_the_model_is_registered_as_version_one(tracked):
    result, _, tmp_path = tracked
    assert result["version"] == "1"
    track_tfidf.use_tracking_dir(tmp_path / "mlruns")
    assert [str(v.version) for v in MlflowClient().search_model_versions(f"name='{track_tfidf.MODEL_NAME}'")] == ["1"]


def reference_bundle(prepared_dir, c=10, scheme="random"):
    frames = dataset.load_split(scheme, prepared_dir)
    vectorizer, classifier = tfidf.fit(pd.concat([frames["train"], frames["val"]], ignore_index=True), c)
    return {"vectorizer": vectorizer, "classifier": classifier}


def test_a_specific_version_can_be_chosen_by_number_or_text(tracked, prepared_dir, tmp_path):
    _, _, tracking = tracked
    for version in (1, "1"):
        info = export_registered_model.export(version, tmp_path / f"out{version}", tracking / "mlruns")
        assert info["model_version"] == "1"
    with pytest.raises(SystemExit):
        export_registered_model.export("9", tmp_path / "none", tracking / "mlruns")


def test_the_export_writes_the_model_the_card_and_the_info(tracked, prepared_dir, tmp_path):
    _, _, tracking = tracked
    reference = tmp_path / "reference.joblib"
    joblib.dump(reference_bundle(prepared_dir), reference)
    out = tmp_path / "out"
    info = export_registered_model.export("latest", out, tracking / "mlruns", compare_with=reference, processed_dir=prepared_dir)

    assert info["model_version"] == "1"
    bundle = joblib.load(out / "tfidf_lr.joblib")
    assert bundle["model_version"] == "1" and bundle["thresholds"] == [0.4, 0.6]
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Version | 1" in card and "results/results.md" in card and "0.767" in card and "fingerprint" in card.lower()
    assert (out / "model_info.json").exists()


def test_the_export_stops_and_writes_nothing_when_the_predictions_differ(tracked, prepared_dir, tmp_path):
    _, _, tracking = tracked
    other = tmp_path / "other.joblib"
    joblib.dump(reference_bundle(prepared_dir, c=0.3), other)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as stopped:
        export_registered_model.export("latest", out, tracking / "mlruns", compare_with=other, processed_dir=prepared_dir)
    assert "STOPPED" in str(stopped.value)
    assert not out.exists()
