"""Export a registered MLflow model version to deploy/model/, the folder that the API and the Docker image use.

    python scripts/export_registered_model.py --version latest --compare models/tfidf_lr.joblib

It writes three files:
    tfidf_lr.joblib   the vectorizer and the classifier, with the classes and the confidence thresholds
    model_info.json   model name, version, run id, data fingerprints (the API reports the version from here)
    MODEL_CARD.md     version, data fingerprints, validation metrics, published test metrics with their source, date

With --compare it first checks that the exported model gives the same predictions as an existing model file on the first
1,000 VALIDATION rows. If it does not, nothing is written and the script stops. The test set is never opened here.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from scripts.track_tfidf import MODEL_NAME, TRACKING_DIR, use_tracking_dir  # noqa: E402
from src import config  # noqa: E402
from src.prepare import dataset  # noqa: E402

DEPLOY_DIR = ROOT / "deploy" / "model"
MODEL_FILE = "tfidf_lr.joblib"


def pick_version(client, version):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"no registered model named {MODEL_NAME}. Run scripts/track_tfidf.py first.")
    if version == "latest":
        return max(versions, key=lambda v: int(v.version))
    chosen = [v for v in versions if str(v.version) == str(version)]
    if not chosen:
        raise SystemExit(f"{MODEL_NAME} has no version {version}")
    return chosen[0]


def build_bundle(pipeline, params, version):
    """The same structure as models/tfidf_lr.joblib, plus the model name and version."""
    classifier = pipeline.named_steps["classifier"]
    return {"vectorizer": pipeline.named_steps["tfidf"], "classifier": classifier, "classes": list(classifier.classes_),
            "thresholds": json.loads(params["confidence_thresholds"]), "scheme": params["scheme"], "C": float(params["C"]),
            "trained_on_rows": int(params["final_model_rows"]), "model_name": MODEL_NAME, "model_version": str(version)}


def probabilities(bundle, texts):
    return bundle["classifier"].predict_proba(bundle["vectorizer"].transform(texts))


def same_predictions(bundle_a, bundle_b, texts):
    """True if both bundles give the same labels and the same probabilities (to 1e-9) on the texts."""
    a, b = probabilities(bundle_a, texts), probabilities(bundle_b, texts)
    same_labels = (np.asarray(bundle_a["classifier"].classes_)[a.argmax(axis=1)] == np.asarray(bundle_b["classifier"].classes_)[b.argmax(axis=1)]).all()
    return bool(same_labels and a.shape == b.shape and np.allclose(a, b, atol=1e-9))


def model_card(version, run, params, metrics, tags, exported_at):
    fingerprints = "\n".join(f"| {name.removeprefix('fingerprint_')} | `{value}` |" for name, value in sorted(params.items()) if name.startswith("fingerprint_"))
    return f"""# Model card: {MODEL_NAME}

TF-IDF (1 to 2 word n-grams) and logistic regression that predicts the vehicle component of an NHTSA complaint text.

| Item | Value |
| --- | --- |
| Registered model | {MODEL_NAME} |
| Version | {version.version} |
| MLflow run | {run.info.run_id} |
| Exported (UTC) | {exported_at} |
| Split used | {params['scheme']} split (train and validation parts, {params['final_model_rows']} complaints) |
| Regularisation C | {params['C']} |
| Confidence thresholds | {params['confidence_thresholds']} (below the first one the API asks for human review) |
| Git commit of the training run | {tags.get('git_commit', 'unknown')} (working tree changed: {tags.get('git_dirty', 'unknown')}) |

## Validation metrics

Scored on the validation part with a model fitted on the training part only. The registered model above was then refitted on training plus validation, so it has no
honest validation score of its own.

| Metric | Value |
| --- | --- |
| Validation accuracy | {metrics['val_accuracy']:.4f} |
| Validation macro-F1 | {metrics['val_macro_f1']:.4f} |

## Published test metrics

Copied from `{tags.get('source', 'results/results.md')}`, not computed by this export. Split: {tags.get('published_test_scheme')}, {tags.get('published_test_items')} test complaints.

| Metric | Value |
| --- | --- |
| Test accuracy | {tags.get('published_test_accuracy')} |
| Test macro-F1 | {tags.get('published_test_macro_f1')} |

## Data fingerprints

A short code for the list of complaint ids of each split, from `results/data_report.json`. The same fingerprints mean the same split.

| Split | Fingerprint |
| --- | --- |
{fingerprints}

## Limits

- Public US data (NHTSA vehicle owner complaints). Labels were assigned by NHTSA and have noise.
- One training run, no search over settings beyond the choice of C on the validation set.
- The fine-tuned encoder scores higher in this project, but its weights were not saved, so this classical model is the one that is served.
- The export did not open any test set.
"""


def export(version="latest", out_dir=DEPLOY_DIR, tracking_dir=TRACKING_DIR, compare_with=None, compare_rows=1000,
           processed_dir=config.PROCESSED_DIR):
    use_tracking_dir(tracking_dir)
    client = MlflowClient()
    chosen = pick_version(client, version)
    run = client.get_run(chosen.run_id)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags

    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{chosen.version}")
    bundle = build_bundle(pipeline, params, chosen.version)

    if compare_with:
        texts = dataset.load_split(params["scheme"], processed_dir)["val"]["text"].head(compare_rows).tolist()
        if not same_predictions(bundle, joblib.load(compare_with), texts):
            raise SystemExit(f"STOPPED: the exported model does not give the same predictions as {compare_with} "
                             f"on the first {len(texts)} validation rows. Nothing was written.")
        print(f"same predictions as {compare_with} on the first {len(texts)} validation rows")

    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle["created"] = exported_at
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / MODEL_FILE, compress=3)
    info = {"model_name": MODEL_NAME, "model_version": str(chosen.version), "run_id": run.info.run_id, "scheme": params["scheme"],
            "exported_utc": exported_at, "fingerprints": {k.removeprefix("fingerprint_"): v for k, v in params.items() if k.startswith("fingerprint_")}}
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (out_dir / "MODEL_CARD.md").write_text(model_card(chosen, run, params, metrics, tags, exported_at), encoding="utf-8")
    print(f"exported {MODEL_NAME} version {chosen.version} to {out_dir} ({(out_dir / MODEL_FILE).stat().st_size:,} bytes)")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="latest", help="a version number or latest")
    parser.add_argument("--out", default=str(DEPLOY_DIR))
    parser.add_argument("--compare", help="an existing model file whose predictions must be the same, e.g. models/tfidf_lr.joblib")
    args = parser.parse_args()
    export(args.version, args.out, compare_with=args.compare)


if __name__ == "__main__":
    main()
