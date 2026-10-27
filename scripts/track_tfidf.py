"""Track the TF-IDF + logistic regression with MLflow. Only the VALIDATION split is scored.

    python scripts/track_tfidf.py --scheme time

What it does:
  1. fits the model exactly as src/baselines does (same settings, same seed, the C that was chosen on the
     validation set in results/tuning/) on the training part and scores it on the VALIDATION part;
  2. fits the same model on training plus validation, which is how the API model models/tfidf_lr.joblib is made;
  3. logs parameters, validation accuracy and macro-F1, the data fingerprints from results/data_report.json and the git
     commit to a local MLflow folder (mlruns/, not committed);
  4. logs the already published test numbers from results/results.md as tags named published_test_*, with the tag
     source=results/results.md. They are copied from that file, never computed here;
  5. registers the model from step 2 in the MLflow model registry as complaint-classifier-tfidf.

The test set is never opened or scored here.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402
from src.baselines import tfidf  # noqa: E402
from src.baselines.run_baseline import tuning_path  # noqa: E402
from src.evaluate import metrics  # noqa: E402
from src.prepare import dataset  # noqa: E402

MODEL_NAME = "complaint-classifier-tfidf"
EXPERIMENT = "complaint-classifier"
TRACKING_DIR = ROOT / "mlruns"
REPORT_FILE = config.RESULTS_DIR / "data_report.json"
RESULTS_FILE = config.RESULTS_DIR / "results.md"
SOURCE_TAG = "results/results.md"


def use_tracking_dir(tracking_dir):
    """Point MLflow at a local folder (a small database for runs and the registry, plus the artifacts)."""
    tracking_dir = Path(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (tracking_dir / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(tracking_dir / "artifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def published_test_numbers(results_text, scheme):
    """The TF-IDF row of the 'All test items' table of one split in results/results.md, copied as text."""
    section = results_text.split(f"## {scheme.capitalize()} split")[1]
    section = section.split("\n## ")[0]
    table = section.split("### All test items")[1].split("\n###")[0]
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "TF-IDF + logistic regression":
            return {"published_test_scheme": scheme, "published_test_items": cells[1],
                    "published_test_accuracy": cells[2], "published_test_macro_f1": cells[3]}
    raise ValueError(f"no TF-IDF row in the {scheme} 'All test items' table")


def data_fingerprints(report):
    """Flat dict like {'fingerprint_time_train': '...'} for every split of the data report."""
    return {f"fingerprint_{scheme}_{part}": info["fingerprint"]
            for scheme, parts in report["splits"].items() for part, info in parts.items()}


def git_info(cwd=ROOT):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def model_params(vectorizer, classifier):
    return {"ngram_range": "-".join(map(str, vectorizer.ngram_range)), "min_df": vectorizer.min_df,
            "max_features": vectorizer.max_features, "sublinear_tf": vectorizer.sublinear_tf, "lowercase": vectorizer.lowercase,
            "C": classifier.C, "max_iter": classifier.max_iter, "random_state": classifier.random_state}


def track(scheme="time", processed_dir=config.PROCESSED_DIR, tuned=None, report=None, results_text=None,
          tracking_dir=TRACKING_DIR, register=True):
    """Run the whole tracking. Returns the run id and the registered model version."""
    tuned = tuned or json.loads(tuning_path("tfidf_lr", scheme).read_text(encoding="utf-8"))
    report = report or json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    results_text = results_text or RESULTS_FILE.read_text(encoding="utf-8")
    best_c = tuned["best_C"]

    frames = dataset.load_split(scheme, processed_dir)          # training and validation parts only
    train, val = frames["train"], frames["val"]
    classes = dataset.class_list(processed_dir)

    # 1. fit on the training part, score on the validation part
    model = tfidf.fit(train, best_c)
    predicted, _ = tfidf.predict(model, val["text"])
    validation = metrics.compute_metrics(val["label"], predicted, classes)

    # 2. the model that is registered and served: training plus validation, like src/baselines/export_model.py
    train_all = pd.concat([train, val], ignore_index=True)
    vectorizer, classifier = tfidf.fit(train_all, best_c)

    commit, dirty = git_info()
    use_tracking_dir(tracking_dir)
    with mlflow.start_run(run_name=f"tfidf_lr_{scheme}") as run:
        mlflow.log_params({"scheme": scheme, **model_params(vectorizer, classifier),
                           "train_rows": len(train), "val_rows": len(val), "final_model_rows": len(train_all),
                           "confidence_thresholds": json.dumps(tuned["confidence_thresholds"]), **data_fingerprints(report)})
        mlflow.log_metrics({"val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"]})
        mlflow.set_tags({"git_commit": commit, "git_dirty": str(dirty), "source": SOURCE_TAG,
                         "validation_fit": "training part only", "registered_model_fit": "training plus validation part",
                         **published_test_numbers(results_text, scheme)})
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
        info = mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=MODEL_NAME if register else None)
    version = getattr(info, "registered_model_version", None)
    version = str(version) if version is not None else None
    return {"run_id": run.info.run_id, "version": version, "validation": validation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=config.SPLIT_SCHEMES, default="time")
    args = parser.parse_args()
    result = track(args.scheme)
    print(f"run {result['run_id']}, registered {MODEL_NAME} version {result['version']}")
    print(f"validation accuracy {result['validation']['accuracy']:.4f}, macro-F1 {result['validation']['macro_f1']:.4f} (training part only)")


if __name__ == "__main__":
    main()
