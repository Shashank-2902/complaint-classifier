"""Run a zero-shot language model on the fixed test subset.

    python -m src.llm.run_llm --provider groq --wait
    python -m src.llm.run_llm --provider zai

One run on the shared 1000-item subset. The answers are scored in both the random-split and the time-split
table, because these items are test items in both splits (see src/evaluate/subsets.py).
The key comes from the environment or from .env (see .env.example). Raw answers are cached in
results/llm_cache/, so the run can be stopped and started again (for example after a daily rate limit).
"""
import argparse
import os
import statistics

from src import config
from src.evaluate.predictions import write_predictions
from src.evaluate.subsets import groq_subset_ids
from src.llm import groq_zero_shot, providers
from src.llm.cache import JsonlCache
from src.prepare import dataset


def load_env_file():
    path = config.ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                if value.strip():
                    os.environ.setdefault(key.strip(), value.strip())


def run(provider, chat, model, max_rpm, wait_seconds=None):
    system = providers.PROVIDERS[provider]["system"]
    classes = dataset.class_list()
    wanted = groq_subset_ids()
    resuming = any((system, scheme) in dataset.used_tests() for scheme in config.SPLIT_SCHEMES)
    subset = dataset.load_test_items_once(config.SPLIT_SCHEMES, system, wanted, force=resuming)

    cache = JsonlCache(config.RESULTS_DIR / "llm_cache" / f"{model.replace('/', '_')}.jsonl")
    items = list(zip(subset["odi_no"], subset["text"]))
    records = groq_zero_shot.classify_until_done(chat, model, items, classes, cache, wait_seconds, max_rpm=max_rpm)

    prompt_tokens = sum(r["prompt_tokens"] for r in records)
    completion_tokens = sum(r["completion_tokens"] for r in records)
    total_cost = groq_zero_shot.cost_usd(model, prompt_tokens, completion_tokens)
    meta = {
        "system": system, "provider": provider, "model": model, "temperature": 0,
        "n_items": len(records), "invalid_answers": sum(1 for r in records if r["pred"] == "INVALID"),
        "latency_ms_per_item": round(1000 * statistics.median(r["latency_s"] for r in records), 1),
        "latency_device": f"{provider} cloud API, includes network time",
        "prompt_tokens_total": prompt_tokens, "completion_tokens_total": completion_tokens,
        "cost_per_1000_usd": round(1000 * total_cost / len(records), 4),
        "cost_note": f"from token counts and the price list of {model}",
    }
    for scheme in config.SPLIT_SCHEMES:
        write_predictions(system, scheme, [r["odi_no"] for r in records], [r["pred"] for r in records],
                          [None] * len(records), {**meta, "scheme": scheme})
    print(f"{provider} / {model}: {len(records)} items, {meta['invalid_answers']} invalid answers, "
          f"cost per 1000 items = ${meta['cost_per_1000_usd']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=list(providers.PROVIDERS), default="groq")
    parser.add_argument("--model", help="override the default model of the provider")
    parser.add_argument("--max-rpm", type=int, default=25, help="requests per minute, keep below your plan's limit")
    parser.add_argument("--wait", action="store_true",
                        help="when a daily token limit is reached, wait 5 minutes and continue (keep the computer awake)")
    args = parser.parse_args()

    load_env_file()
    settings = providers.PROVIDERS[args.provider]
    if not os.getenv(settings["key_variable"]):
        raise SystemExit(f"{settings['key_variable']} is not set. Put it in .env (see .env.example).")
    model = args.model or providers.model_for(args.provider)
    run(args.provider, providers.make_chat(args.provider, model), model, args.max_rpm, 300 if args.wait else None)


if __name__ == "__main__":
    main()
