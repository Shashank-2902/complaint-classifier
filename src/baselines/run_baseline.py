"""Train, tune and evaluate the two baselines.

    python -m src.baselines.run_baseline --system tfidf_lr --scheme random --stage tune
    python -m src.baselines.run_baseline --system tfidf_lr --scheme random --stage final

Stage "tune" uses only the training and validation parts. It picks the best C on the validation
set and saves it to results/tuning/. Stage "final" trains with that C and predicts the test set,
which it opens exactly once (see src/prepare/dataset.py).
"""
import argparse
import json
import platform

from src import config
from src.baselines import embeddings, tfidf
from src.evaluate import metrics
from src.evaluate.predictions import write_predictions
from src.prepare import dataset


def tuning_path(system, scheme):
    return config.TUNING_DIR / f"{system}__{scheme}.json"


def split_vectors(frames, all_frame, encoder=None):
    """For the embedding system, pick the cached vectors that belong to each part of a split."""
    vectors = embeddings.embed_dataset(all_frame, encoder)
    position = {odi: i for i, odi in enumerate(all_frame["odi_no"])}
    return {name: vectors[[position[o] for o in part["odi_no"]]] for name, part in frames.items()}


def tune(system, scheme, encoder=None):
    frames = dataset.load_split(scheme)
    train, val = frames["train"], frames["val"]

    if system == "tfidf_lr":
        best_c, grid = tfidf.tune(train, val)
        model = tfidf.fit(train, best_c)
        predicted, confidence = tfidf.predict(model, val["text"])
    else:
        vectors = split_vectors(frames, dataset.load_all(), encoder)
        best_c, grid = embeddings.tune(vectors["train"], train["label"], vectors["val"], val["label"])
        classifier = embeddings.make_classifier(best_c).fit(vectors["train"], train["label"])
        predicted, confidence = embeddings.predict(classifier, vectors["val"])

    thresholds = metrics.choose_thresholds(confidence)
    result = {"system": system, "scheme": scheme, "best_C": best_c, "grid": grid, "confidence_thresholds": thresholds,
              "val_metrics": metrics.compute_metrics(val["label"], predicted, dataset.class_list())}
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    tuning_path(system, scheme).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{system} / {scheme}: best C = {best_c}, validation macro-F1 = {result['val_metrics']['macro_f1']:.3f}")
    return result


def final(system, scheme, force=False, encoder=None):
    tuned = json.loads(tuning_path(system, scheme).read_text(encoding="utf-8"))
    frames = dataset.load_split(scheme)
    train = frames["train"]
    test = dataset.load_test_once(scheme, system, force=force)

    if system == "tfidf_lr":
        model = tfidf.fit(train, tuned["best_C"])
        predicted, confidence = tfidf.predict(model, test["text"])
        latency = tfidf.measure_latency(model, test["text"].tolist())
        model_name = "TfidfVectorizer(1-2 grams, 100k features) + LogisticRegression"
    else:
        all_frame = dataset.load_all()
        vectors = split_vectors(frames, all_frame, encoder)
        classifier = embeddings.make_classifier(tuned["best_C"]).fit(vectors["train"], train["label"])
        position = {odi: i for i, odi in enumerate(all_frame["odi_no"])}
        cached = embeddings.embed_dataset(all_frame, encoder)
        predicted, confidence = embeddings.predict(classifier, cached[[position[o] for o in test["odi_no"]]])
        latency = embeddings.measure_latency(encoder or embeddings.load_encoder(), classifier, test["text"].tolist(), n=100)
        model_name = f"{embeddings.MODEL_NAME} + LogisticRegression"

    meta = {"system": system, "scheme": scheme, "model": model_name, "best_C": tuned["best_C"],
            "latency_ms_per_item": round(latency, 2), "latency_device": f"CPU ({platform.processor() or platform.machine()})",
            "cost_per_1000_usd": 0.0, "cost_note": "runs locally, no API cost", "n_items": len(test)}
    write_predictions(system, scheme, test["odi_no"], predicted, confidence, meta)
    result = metrics.compute_metrics(test["label"], predicted, dataset.class_list())
    print(f"{system} / {scheme}: test accuracy = {result['accuracy']:.3f}, macro-F1 = {result['macro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["tfidf_lr", "embedding_lr"], required=True)
    parser.add_argument("--scheme", choices=config.SPLIT_SCHEMES, required=True)
    parser.add_argument("--stage", choices=["tune", "final"], required=True)
    parser.add_argument("--force", action="store_true", help="repeat the final run (it is logged as forced)")
    args = parser.parse_args()
    if args.stage == "tune":
        tune(args.system, args.scheme)
    else:
        final(args.system, args.scheme, force=args.force)


if __name__ == "__main__":
    main()
