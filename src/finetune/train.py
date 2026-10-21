"""Fine-tune a small encoder (DistilBERT by default) to classify complaints.

    python -m src.finetune.train --scheme random
    python -m src.finetune.train --scheme time

Run it on a free GPU (Colab or Kaggle, see src/finetune/colab_finetune.ipynb). On a laptop CPU it is
too slow for the full data. Use --smoke to check that the code runs: it trains on a few hundred
complaints and evaluates on validation rows, never on the test set.

What happens: the model starts from a pretrained language model and is trained for a few epochs on the
training set. After every epoch it is scored on the validation set, and the best epoch is kept.
The test set is opened once, at the end.
"""
import argparse
import copy
import platform
import time

import numpy as np

from src import config
from src.evaluate import metrics
from src.evaluate.predictions import write_predictions
from src.finetune import data
from src.prepare import dataset

DEFAULT_MODEL = "distilbert-base-uncased"


def tokenize(tokenizer, texts, max_len):
    return tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def predict_logits(model, tokenizer, texts, device, max_len, batch_size=64):
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in data.make_batches(len(texts), batch_size):
            encoded = tokenize(tokenizer, [texts[i] for i in batch], max_len).to(device)
            outputs.append(model(**encoded).logits.float().cpu().numpy())
    return np.vstack(outputs)


def measure_latency(model, tokenizer, texts, device, max_len, n=200):
    """Median milliseconds to classify one text at a time."""
    import torch

    times = []
    for text in texts[:n]:
        started = time.perf_counter()
        predict_logits(model, tokenizer, [text], device, max_len, batch_size=1)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
    return float(np.median(times))


def train_model(train, val, classes, args, device):
    """Train and return (model, tokenizer, epoch table). The best epoch on the validation set is kept."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(config.SEED)
    to_id, _ = data.label_maps(classes)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(classes)).to(device)

    texts = train["text"].tolist()
    labels = torch.tensor(data.encode_labels(train["label"], to_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(texts) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)

    best_state, best_f1, table = None, -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in data.make_batches(len(texts), args.batch_size, seed=config.SEED + epoch):
            encoded = tokenize(tokenizer, [texts[i] for i in batch], args.max_len).to(device)
            loss = model(**encoded, labels=labels[batch].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        logits = predict_logits(model, tokenizer, val["text"].tolist(), device, args.max_len)
        _, to_label = data.label_maps(classes)
        predicted, _ = data.to_predictions(logits, to_label)
        score = metrics.compute_metrics(val["label"], predicted, classes)
        table.append({"epoch": epoch, "val_macro_f1": score["macro_f1"], "val_accuracy": score["accuracy"]})
        print(f"epoch {epoch}: validation macro-F1 = {score['macro_f1']:.3f}, accuracy = {score['accuracy']:.3f}", flush=True)
        if score["macro_f1"] > best_f1:
            best_f1, best_state = score["macro_f1"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, tokenizer, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=config.SPLIT_SCHEMES, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the code, never touches the test set")
    parser.add_argument("--force", action="store_true", help="repeat the final run (it is logged as forced)")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = dataset.class_list()
    frames = dataset.load_split(args.scheme)
    train, val = frames["train"], frames["val"]

    if args.smoke:
        train, val = train.head(300), val.head(100)
        args.epochs = 1
        args.max_len = 64
    print(f"device: {device}, train rows: {len(train)}, validation rows: {len(val)}", flush=True)

    model, tokenizer, table = train_model(train, val, classes, args, device)

    # The test set is opened here, once. In smoke mode the validation rows stand in for it.
    test = val if args.smoke else dataset.load_test_once(args.scheme, "finetuned_encoder", force=args.force)
    logits = predict_logits(model, tokenizer, test["text"].tolist(), device, args.max_len)
    _, to_label = data.label_maps(classes)
    predicted, confidence = data.to_predictions(logits, to_label)
    latency = measure_latency(model, tokenizer, test["text"].tolist(), device, args.max_len)

    meta = {"system": "finetuned_encoder", "scheme": args.scheme, "model": args.model, "epochs": args.epochs,
            "learning_rate": args.lr, "batch_size": args.batch_size, "max_len": args.max_len, "epoch_table": table,
            "latency_ms_per_item": round(latency, 2), "latency_device": f"{device} ({platform.processor() or platform.machine()})",
            "cost_per_1000_usd": 0.0, "cost_note": "trained on a free GPU, runs locally, no API cost",
            "n_items": len(test), "smoke_test": args.smoke}
    directory = config.RESULTS_DIR / "smoke" if args.smoke else config.PREDICTIONS_DIR
    write_predictions("finetuned_encoder", args.scheme, test["odi_no"], predicted, confidence, meta, directory)
    result = metrics.compute_metrics(test["label"], predicted, classes)
    label = "validation (smoke test)" if args.smoke else "test"
    print(f"{label}: accuracy = {result['accuracy']:.3f}, macro-F1 = {result['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
