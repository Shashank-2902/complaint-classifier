# complaint-classifier

Which part of the car is a complaint about? This project predicts the vehicle component from the free text of real
vehicle owner complaints filed with NHTSA, and compares four kinds of systems on the same data: a TF-IDF baseline, sentence
embeddings, a fine-tuned small encoder and a zero-shot language model on Groq.

The point is the comparison, not the highest number. A strong classical baseline can win, and the project reports it honestly,
including cost, speed and the cases where each system fails.

- **Data:** public NHTSA complaint data, 2015 to October 2026. Source, format and terms are in [data/README.md](data/README.md).
- **Label:** the top-level NHTSA component (14 most frequent classes plus `OTHER`). **The label has noise**: it was assigned by NHTSA staff
  or chosen on the complaint form, so accuracy here is agreement with that label, not with absolute truth.
- **Not included:** RAG, a chatbot, a web UI.

## Results

All numbers are measured on test sets that were opened once per system. Any system that has not been run says "not run".
Full tables, F1 per class and the confidence thresholds are in [results/results.md](results/results.md).

Two splits: a **random** split (easy: test complaints look like the training ones) and a **time** split (train on the oldest 70 percent,
test on the newest 20 percent, 2024-07-29 to 2026-10-18).

**Full test sets (13,840 complaints each)**

| System | Random: accuracy | Random: macro-F1 | Time: accuracy | Time: macro-F1 | Latency per item | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.767 | 0.742 | 0.713 | 0.674 | 2 ms | 0 (local) |
| Sentence embeddings + logistic regression | 0.756 | 0.736 | 0.705 | 0.662 | 23 to 25 ms | 0 (local) |
| Fine-tuned small encoder (DistilBERT) | not run | not run | not run | not run | not run | not run |

**The fixed subset of 1,000 test items (the same items for every system, test items in both splits)**

| System | Random: accuracy | Random: macro-F1 | Time: accuracy | Time: macro-F1 | Latency per item | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.775 | 0.745 | 0.735 | 0.678 | 2 ms | 0 (local) |
| Sentence embeddings + logistic regression | 0.769 | 0.727 | 0.715 | 0.654 | 23 to 25 ms | 0 (local) |
| Fine-tuned small encoder (DistilBERT) | not run | not run | not run | not run | not run | not run |
| Z.ai zero-shot (`glm-4.5-flash`, free) | 0.565 | 0.592 | 0.565 | 0.592 | 719 ms | $0 (free plan) |
| Groq zero-shot (`openai/gpt-oss-20b`) | 0.645 | 0.629 | 0.645 | 0.629 | 2422 ms | $0.067 |

The zero-shot models are not trained, so they give the same answers for both splits, scored on the same 1,000 items. Their latency includes the network and any wait after a rate limit. Groq returned 43 answers (4.3 percent) that were not one of the 15 classes; they count as wrong.
Latency of the local systems is the median time for one complaint on this laptop CPU (Intel, no GPU), including the embedding step.

### What the results say

- **The classical baseline wins so far.** TF-IDF with logistic regression is as good as or better than the embedding model, and it is about 10 times faster.
- **Zero-shot is clearly worse here.** On the same 1,000 items, Groq `gpt-oss-20b` scores 0.645 accuracy and GLM-4.5-Flash 0.565, against 0.775 (random split) and 0.735 (time split) for TF-IDF. Groq costs $0.067 per 1000 items, TF-IDF nothing. They understand the complaints but cannot know NHTSA's labelling rules,
  for example when staff use `ENGINE` and when `ENGINE AND ENGINE COOLING`, or what `OTHER` contains. It does better than TF-IDF only on classes whose frequency changed a lot over time
  (`AIR BAGS`, `FORWARD COLLISION AVOIDANCE`), with only 36 and 19 items.
- **The time split is harder.** Accuracy drops by about 5 points and macro-F1 by about 7 points on newer complaints. NHTSA's component use changes over time:
  air bag complaints are 15.5 percent of the training part and 3.5 percent of the test part, `ENGINE AND ENGINE COOLING` goes from 0.7 to 3.5 percent, and
  `FORWARD COLLISION AVOIDANCE` hardly existed before 2020. The F1 of `ENGINE AND ENGINE COOLING` falls to 0.29 on the time split.
- **Macro-F1 is lower than accuracy** because the small and vague classes are harder (`VEHICLE SPEED CONTROL` 0.45 to 0.52, `OTHER` about 0.63).
- **Abstaining on low confidence works.** The thresholds are chosen on the validation set (they would accept about 70 and 40 percent of validation items).
  On the time-split test set, TF-IDF answers 68 percent of the items with 83.4 percent accuracy at a confidence of 0.718, and 38.6 percent of the items with
  92.0 percent accuracy at 0.949. The rest would go to a human. On the random split: 88.2 percent accuracy at 70.3 percent coverage, 95.7 percent at 39.7 percent.
- **Most errors are not clear model mistakes.** 61 to 64 percent of the errors involve the catch-all class `OTHER`, and 9.6 percent of the time-split errors are mix-ups between
  `ENGINE` and `ENGINE AND ENGINE COOLING`, two NHTSA components with similar content. Reading real examples, the model is clearly wrong in about two of seven groups of mistakes,
  and the label is arguable in most of the others. See [results/error_analysis.md](results/error_analysis.md) (these judgements are my own reading, not a measurement).

### The row that is not run yet

- **Fine-tuned encoder.** It needs a GPU, which this project's laptop does not have. The training script and a Colab/Kaggle notebook are ready
  ([src/finetune/](src/finetune/)). Start it yourself, see below. Until its predictions are added to `results/predictions/`, the row says "not run".

The Groq run needed several days because the free plan allows 200,000 tokens per day for this model, about 440 answers per day. The answers are cached
and the run could be stopped and started again.

## How the comparison is kept fair

- **Duplicates are removed before splitting** (exact copies and near copies), so the same complaint cannot be in training and test.
- **The test set is opened once per system and split.** Tuning code receives only the training and validation parts. The test rows come from one function that logs every access to
  `results/test_access_log.jsonl` and refuses a second use. A test checks this.
- **Settings are chosen on the validation set only.** The confidence thresholds are chosen on validation too.
- **Fixed seeds** and the same test items for every system. The Groq row runs on a fixed subset of 1,000 test items that are test items in **both** splits, so one run fills both tables.
  Every system is also reported on exactly those 1,000 items in `results/results.md`.
- **Macro-F1 next to accuracy**, because the classes are imbalanced.
- **Multi-component complaints are dropped** (218,327 complaints, 28.1 percent), because their label is not clear. See [data/README.md](data/README.md) for every step and count.
- **Nothing is invented.** The results table is built from prediction files by a script. A system without a prediction file is shown as "not run".

## Run it

Python 3.11. For the tests you need only `pip install -r requirements-dev.txt`.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt            # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest                              # 87 tests, no GPU, no Groq key, no download needed
```

The full experiment (about 190 MB download, a few hours on a laptop):

```bash
.venv/Scripts/pip install -r requirements-ml.txt             # torch, sentence-transformers, transformers
.venv/Scripts/python -m src.prepare.download                 # NHTSA files into data/raw/
.venv/Scripts/python -m src.prepare.run                      # clean, label, dedupe, sample, split (about 10 minutes)

# the two baselines: tune on validation, then one final run on the test set
python -m src.baselines.run_baseline --system tfidf_lr --scheme random --stage tune
python -m src.baselines.run_baseline --system tfidf_lr --scheme random --stage final
#   ... same for --scheme time, and for --system embedding_lr

python -m src.evaluate.report                                # writes results/results.md
python -m src.evaluate.error_analysis --system tfidf_lr --scheme time
```

**Zero-shot language models** need a free key in `.env` (copy `.env.example`). Check the current model names and rate limits in each provider's console.

```bash
python -m src.llm.run_llm --provider groq --wait   # Groq gpt-oss-20b. --wait: at the daily token limit, wait and continue (keep the computer awake)
python -m src.llm.run_llm --provider zai           # Z.ai glm-4.5-flash (free), one request at a time
```

Every raw answer is cached in `results/llm_cache/`, so a stopped run resumes without new cost. The whole run is about 450,000 tokens, roughly 6 US cents at the listed price.

**Fine-tuned encoder** (you start it, it uses your free GPU quota): open `src/finetune/colab_finetune.ipynb` in Google Colab (GPU runtime) or Kaggle (GPU on, internet on) and run all cells.
It prepares the data with the same seed, **checks that the splits are identical to this repository's**, trains DistilBERT for both splits (typically 10 to 20 minutes per split on a T4, an estimate that I could not measure without a GPU) and gives you
`finetune_results.zip`. Unzip it into `results/`, then run `python -m src.evaluate.report`.

**The API** (serves the TF-IDF model, trained on the training and validation parts of the time split):

```bash
python -m src.baselines.export_model --scheme time
uvicorn api.main:app --port 8000                              # then open http://localhost:8000/docs
```

`POST /classify` with `{"text": "..."}` returns the label, the confidence, the top three classes and `needs_human_review`.
Below the confidence threshold (0.718 by default, or send your own) the answer is "needs human review".

## Project layout

```
data/                data/README.md has the source, format, terms and every preparation step
src/prepare/         download, load, clean, labels, dedupe, splits, the guarded test access
src/baselines/       TF-IDF and embedding systems, tuning and final runs, model export for the API
src/finetune/        training script and the Colab/Kaggle notebook
src/llm/             zero-shot prompt, answer parser, cache, rate limit handling, providers (Groq, Z.ai), run script
src/evaluate/        metrics, confusion analysis, thresholds, results table
api/                 FastAPI /classify with the abstain option
tests/               86 tests, a small real-data file (tests/fixtures), no GPU or key needed
docs/learn/          concept notes with walkthroughs, interview questions and glossaries
results/             results.md, error analysis, predictions, tuning files, access log, Groq cache
```

## Limits and things to know

- **Label noise.** Scores are agreement with NHTSA's label. Two reasonable people can pick different components for the same defect.
- **Taxonomy and drift.** NHTSA has two similar engine components, and the class mix changes over the years. The time split shows how much this matters.
- **Sampled data.** The experiments use a random sample of 70,000 of the 554,344 unique single-component complaints, so that they run on a laptop. Results on all data could differ slightly.
- **`OTHER` is not one thing.** It mixes NHTSA's own catch-all and every component outside the 14 named classes, so it is hard to predict by design.
- **Complaints can name the component themselves.** Some texts include the recall's component name (for example "(Engine and Engine Cooling)"). I did not remove this, because it is part of the real data.
- **Zero-shot on this task.** The prompt contains only the class names, and a free-plan model was used (`glm-4.5-flash`, 5 of 1,000 answers were not a valid class). It cannot know NHTSA's labelling habits, such as when staff use `ENGINE` and when `ENGINE AND ENGINE COOLING`.
- **Latency is measured on one laptop CPU**, so absolute values will differ elsewhere. The Groq latency will include network time.
- **A discarded first run.** My first full run read the NHTSA files with the wrong text encoding (garbled quotes, for example "doesnât"). I found this while reading real examples, fixed it
  (the files are UTF-8), and ran everything again from scratch. No decision was based on the first run's results, and its test-access log is kept in `results/discarded_run1/` for transparency.
  A test now checks the decoding.

## Data source and licence

NHTSA vehicle owner complaints, https://www.nhtsa.gov/nhtsa-datasets-and-apis. NHTSA's Terms of Use say that information on the site is public information that may be distributed or copied, and that NHTSA
makes no guarantee of accuracy. Details and the exact wording are in [data/README.md](data/README.md). The raw data is not committed.

Code: MIT, see `LICENSE`.
