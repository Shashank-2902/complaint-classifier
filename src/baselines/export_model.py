"""Train the model that the API serves and save it to models/.

    python -m src.baselines.export_model --scheme time

It uses the TF-IDF + logistic regression system with the C that was chosen on the validation set,
trained on the training AND validation parts. The test set is not used. The two confidence
thresholds from the tuning step are stored with the model.
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src import config
from src.baselines import tfidf
from src.baselines.run_baseline import tuning_path
from src.prepare import dataset

MODEL_DIR = config.ROOT / "models"


def export(scheme, directory=MODEL_DIR):
    tuned = json.loads(tuning_path("tfidf_lr", scheme).read_text(encoding="utf-8"))
    frames = dataset.load_split(scheme)
    train = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    vectorizer, classifier = tfidf.fit(train, tuned["best_C"])

    bundle = {"vectorizer": vectorizer, "classifier": classifier, "classes": list(classifier.classes_),
              "thresholds": tuned["confidence_thresholds"], "scheme": scheme, "C": tuned["best_C"],
              "trained_on_rows": len(train), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / "tfidf_lr.joblib")
    print(f"saved {directory / 'tfidf_lr.joblib'} (trained on {len(train)} complaints)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=config.SPLIT_SCHEMES, default="time")
    export(parser.parse_args().scheme)


if __name__ == "__main__":
    main()
