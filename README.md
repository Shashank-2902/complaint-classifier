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

All numbers are measured on the test sets. Every access to a test set is logged, see "How the comparison is kept fair" for what the log shows. All five systems have predictions. A system without a prediction file would be shown as "not run".
Full tables, F1 per class and the confidence thresholds are in [results/results.md](results/results.md).

Two splits: a **random** split (easy: test complaints look like the training ones) and a **time** split (train on the oldest 70 percent,
test on the newest 20 percent, 2024-07-29 to 2026-10-18).

**Full test sets (13,840 complaints each)**

| System | Random: accuracy | Random: macro-F1 | Time: accuracy | Time: macro-F1 | Latency per item | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.767 | 0.742 | 0.713 | 0.674 | 2 ms | 0 (local) |
| Sentence embeddings + logistic regression | 0.756 | 0.736 | 0.705 | 0.662 | 23 to 25 ms | 0 (local) |
| Fine-tuned small encoder (DistilBERT) | 0.803 | 0.790 | 0.781 | 0.764 | 5.6 to 5.9 ms (GPU) | 0 (free GPU) |

**The fixed subset of 1,000 test items (the same items for every system, test items in both splits)**

| System | Random: accuracy | Random: macro-F1 | Time: accuracy | Time: macro-F1 | Latency per item | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.775 | 0.745 | 0.735 | 0.678 | 2 ms | 0 (local) |
| Sentence embeddings + logistic regression | 0.769 | 0.727 | 0.715 | 0.654 | 23 to 25 ms | 0 (local) |
| Fine-tuned small encoder (DistilBERT) | 0.820 | 0.784 | 0.787 | 0.746 | 5.6 to 5.9 ms (GPU) | 0 (free GPU) |
| Z.ai zero-shot (`glm-4.5-flash`, free) | 0.565 | 0.592 | 0.565 | 0.592 | 719 ms | $0 (free plan) |
| Groq zero-shot (`openai/gpt-oss-20b`) | 0.645 | 0.629 | 0.645 | 0.629 | 2422 ms | $0.067 |

The zero-shot models are not trained, so they give the same answers for both splits, scored on the same 1,000 items. Their latency includes the network and any wait after a rate limit. Groq returned 43 answers (4.3 percent) that were not one of the 15 classes; they count as wrong.
Latency of the TF-IDF and embedding systems is the median time for one complaint on this laptop CPU (Intel, no GPU), including the embedding step.
The fine-tuned encoder's latency was measured on a GPU in Colab (the median of single-text calls, see `src/finetune/train.py`), so it is not comparable with the CPU rows.

### What the results say

- **The fine-tuned encoder is the strongest system here.** On the full test sets it scores 0.803 accuracy and 0.790 macro-F1 on the random split and 0.781 and 0.764 on the time split,
  against 0.767 and 0.742, and 0.713 and 0.674, for TF-IDF. Among the systems that run on a laptop CPU, TF-IDF with logistic regression is as good as or better than the embedding model, and it is about 10 times faster.
  On the same test items, the fine-tuned encoder is right and TF-IDF wrong on 1,144 complaints, and the other way round on 642, on the random split (exact McNemar p = 7.8 x 10^-33).
  On the time split the counts are 1,595 and 646 (p = 5.2 x 10^-92). So the difference is not chance on this test set. It says nothing about other data, see [results/paired_test.md](results/paired_test.md).
- **Zero-shot is clearly worse here.** On the same 1,000 items, Groq `gpt-oss-20b` scores 0.645 accuracy and GLM-4.5-Flash 0.565, against 0.775 (random split) and 0.735 (time split) for TF-IDF and 0.820 and 0.787 for the fine-tuned encoder. Groq costs $0.067 per 1000 items, TF-IDF nothing. They understand the complaints but cannot know NHTSA's labelling rules,
  for example when staff use `ENGINE` and when `ENGINE AND ENGINE COOLING`, or what `OTHER` contains. It does better than TF-IDF only on classes whose frequency changed a lot over time
  (`AIR BAGS`, `FORWARD COLLISION AVOIDANCE`), with only 36 and 19 items.
- **The time split is harder, and it hurts the fine-tuned encoder least.** From the random to the time split (full test sets), accuracy falls by 5.4 points for TF-IDF (0.767 to 0.713), 5.1 for the embeddings (0.756 to 0.705)
  and 2.2 for the fine-tuned encoder (0.803 to 0.781). Macro-F1 falls by 6.8 points for TF-IDF (0.742 to 0.674), 7.4 for the embeddings (0.736 to 0.662) and 2.6 for the fine-tuned encoder (0.790 to 0.764).
  The zero-shot models are not trained, so they have no drop (the same 1,000 items are scored for both splits). NHTSA's component use changes over time:
  air bag complaints are 15.5 percent of the training part and 3.5 percent of the test part, `ENGINE AND ENGINE COOLING` goes from 0.7 to 3.5 percent, and
  `FORWARD COLLISION AVOIDANCE` hardly existed before 2020. The F1 of `ENGINE AND ENGINE COOLING` falls from 0.64 to 0.29 for TF-IDF on the time split and from 0.77 to 0.76 for the fine-tuned encoder.
- **Macro-F1 is lower than accuracy** because the small and vague classes are harder (for TF-IDF, `VEHICLE SPEED CONTROL` 0.45 to 0.52, `OTHER` about 0.63).
- **Abstaining on low confidence works.** The thresholds are chosen on the validation set (they would accept about 70 and 40 percent of validation items).
  On the time-split test set, TF-IDF answers 68 percent of the items with 83.4 percent accuracy at a confidence of 0.718, and 38.6 percent of the items with
  92.0 percent accuracy at 0.949. The rest would go to a human. On the random split: 88.2 percent accuracy at 70.3 percent coverage, 95.7 percent at 39.7 percent.
- **Most errors are not clear model mistakes.** 61 to 64 percent of the errors involve the catch-all class `OTHER`, and 9.6 percent of the time-split errors are mix-ups between
  `ENGINE` and `ENGINE AND ENGINE COOLING`, two NHTSA components with similar content. Reading real examples, the model is clearly wrong in about two of seven groups of mistakes,
  and the label is arguable in most of the others. See [results/error_analysis.md](results/error_analysis.md) (these judgements are my own reading, not a measurement).

### Fine-tuning

The model is `distilbert-base-uncased`, trained for 3 epochs with learning rate 5e-5, batch size 32 and a maximum text length of 256 tokens (the settings in
`results/predictions/finetuned_encoder__*.meta.json`). After every epoch the model is scored on the validation set, and the epoch with the highest validation
macro-F1 is kept: its weights are restored before the test set is opened (`src/finetune/train.py`). In both runs this was epoch 3 (validation macro-F1 0.794 on the random split, 0.781 on the time split).
The validation score was still rising at epoch 3, so a longer run might score higher. I did not try it.
There was one training run per split, no search over settings and one seed (42 for the torch seed; the batch order uses 42 plus the epoch number).
The latency (5.6 ms on the random split, 5.9 ms on the time split) is the median of single-text calls on a GPU, so it is not comparable with the CPU rows. The cost is zero because the training ran on a free GPU.

### The fine-tuning runs

I want to be open about how these numbers came about.

- The numbers in the tables come from a second Colab session on 25 Oct 2026 with a T4 GPU. The access log has one entry per split for it, not forced (08:31 and 09:38 UTC).
- A first Colab session, on another account and earlier, also trained the random split. Its screen showed validation epoch 2 macro-F1 0.786 and accuracy 0.796, epoch 3 0.794 and 0.803, and test accuracy 0.803 and macro-F1 0.790.
  Its output was not saved. For its time split I do not know whether it reached the test step, because the session disconnected.
- The access log cannot see that first session, because every Colab session starts from a fresh clone of the repository. So the log does not cover every session: the random test set was opened in the first session as well, and the time test set possibly.
- The second session gave validation epoch 2 macro-F1 0.788 and accuracy 0.797 (see the meta file), and the same epoch 3 and test numbers to three decimals.
- The notebook's data download failed the split check in Colab, because NHTSA updates its files daily and the data of this repository was downloaded on 2026-10-20. I fixed it by uploading `data/processed/dataset.csv.gz` and checking its
  fingerprints against `results/data_report.json`. The notebook now does this (see "Run it").

The Groq run needed several days because the free plan allows 200,000 tokens per day for this model, about 440 answers per day. The answers are cached
and the run could be stopped and started again.

## How the comparison is kept fair

- **Duplicates are removed before splitting** (exact copies and near copies), so the same complaint cannot be in training and test.
- **Every access to a test set is logged.** Tuning code receives only the training and validation parts. The test rows come from one function that writes every access to
  `results/test_access_log.jsonl` and refuses a second normal use of the same system and split. A resumed run is allowed through with `forced = true`. A test checks the refusal.
  What the log shows (times are UTC):

  | System | Accesses per split | Forced | Rows read | Dates |
  | --- | --- | --- | --- | --- |
  | TF-IDF + logistic regression | 1 (random), 1 (time) | 0 | 13,840 | 21 Oct 2026 |
  | Sentence embeddings + logistic regression | 1 (random), 1 (time) | 0 | 13,840 | 21 Oct 2026 |
  | Z.ai GLM-4.5-Flash | 1 (random), 1 (time) | 0 | 1,000 | 21 Oct 2026 |
  | Groq gpt-oss-20b | 7 (random), 7 (time) | 6 per split | 13,840 at the first access, then 1,000 | 21, 22, 23 and 24 Oct 2026 |
  | Fine-tuned encoder (second Colab session) | 1 (random), 1 (time) | 0 | 13,840 | 25 Oct 2026 (08:31 and 09:38) |

  The Groq run was resumed several times over four days because the free plan limits the tokens per day. The first access (21 Oct, 12:35 and 12:50) read the full test sets
  of 13,840 rows. After that the run was restricted to the fixed subset of 1,000 items, and each of the six resumes (forced) read the same 1,000 items again and reused the cached answers.
  So the Groq test items were read more than once, but the same prompt was used every time: `src/llm/prompt.py` has one commit (`6092d74`, the first commit of this repository) and
  `git diff 6092d74 HEAD --stat -- src/llm/prompt.py` prints nothing, so it is identical in the first and the last commit.
  The discarded first run (see below) has its own log: TF-IDF and embeddings on 20 Oct, and Groq on the time split on 20 Oct with one forced access on 21 Oct, all 13,840 rows.
  The log does not see every session: each Colab session starts from a fresh clone with an empty log, and an earlier Colab session also trained the random split (see "The fine-tuning runs").
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
.venv/Scripts/python -m pytest                              # 116 tests, no GPU, no Groq key, no download needed
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
python -m src.evaluate.paired_test                           # fine-tuned against TF-IDF, writes results/paired_test.md
python -m src.evaluate.error_analysis --system tfidf_lr --scheme time
```

**Zero-shot language models** need a free key in `.env` (copy `.env.example`). Check the current model names and rate limits in each provider's console.

```bash
python -m src.llm.run_llm --provider groq --wait   # Groq gpt-oss-20b. --wait: at the daily token limit, wait and continue (keep the computer awake)
python -m src.llm.run_llm --provider zai           # Z.ai glm-4.5-flash (free), one request at a time
```

Every raw answer is cached in `results/llm_cache/`, so a stopped run resumes without new cost. The whole run is about 450,000 tokens, roughly 6 US cents at the listed price.

**Fine-tuned encoder** (it uses a free GPU quota): open `src/finetune/colab_finetune.ipynb` in Google Colab (GPU runtime) and run the cells in order, each one once.
NHTSA updates its files every day, so the notebook does not download the data again. You upload `data/processed/dataset.csv.gz` from your local copy, and the notebook checks its fingerprints against `results/data_report.json`
before it trains. If they differ, it stops. Training DistilBERT for one split took longer than 30 minutes in the second session. At the end the notebook gives you `finetune_results.zip`. Unzip it in the repository folder
(the paths inside the zip already start with `results/`), then run `python -m src.evaluate.report`. I could not run the notebook while editing it, so the current version is untested.

**The API** (serves the TF-IDF model, trained on the training and validation parts of the time split). The model comes from a folder, `deploy/model` by default (set `MODEL_DIR` to use another one):

```bash
python scripts/track_tfidf.py --scheme time                  # fit, score the VALIDATION split only, log to a local MLflow folder (mlruns/), register the model
python scripts/export_registered_model.py --compare models/tfidf_lr.joblib   # write deploy/model/ and MODEL_CARD.md
uvicorn api.main:app --port 8000                              # then open http://localhost:8000/docs
```

`POST /classify` with `{"text": "..."}` returns the label, the confidence, the top three classes, `needs_human_review` and the `model_version`.
Below the confidence threshold (0.718 by default, or send your own) the answer is "needs human review". A text longer than 5,000 characters gets a 422.
`GET /health` returns the status and the `model_version`. `scripts/track_tfidf.py` never scores the test set: it logs the published test numbers from `results/results.md` as tags, with their source.
`python -m src.baselines.export_model --scheme time` still writes `models/tfidf_lr.joblib`, which the export script uses to check that it gives the same predictions.

## Deployment on Azure

**Nothing is deployed.** This section describes how the API can be deployed. The Docker image was built and run on localhost (`docs/deploy_local_check.md`). The Terraform files pass `terraform validate`.
The deploy workflow and the Terraform apply were not run.

The API runs in a Docker image (Python 3.11 slim, non-root user, the model baked in from `deploy/model`) on Azure Container Apps. Terraform (`infra/terraform/`) describes a resource group, a container registry,
a log workspace with a small daily quota, a Container Apps environment and the container app with external HTTPS ingress, 0.25 vCPU and 0.5 Gi, and 0 to 1 replicas.
The app pulls the image from the registry with a managed identity that has the AcrPull role, so there is no registry password. The model is tracked with MLflow on the laptop: a script registers it and another exports a
registered version to `deploy/model`, together with a model card that lists the version, the data fingerprints and the metrics. The CI builds the image and checks `/health`, and validates the Terraform files without credentials.
A separate workflow (`.github/workflows/deploy.yml`, tags `v*` or a manual start) would test, build, log in to Azure with OIDC, push the image and update the app.

Run the image locally:

```bash
docker build -t complaint-classifier:local .
docker run -d --name cc-local -p 127.0.0.1:8000:8000 complaint-classifier:local
curl http://127.0.0.1:8000/health
```

Deploy and tear down with the scripts (they ask you to type a confirmation first, and need the Azure CLI, Terraform and Docker):

```powershell
pwsh ./infra/deploy.ps1
pwsh ./infra/teardown.ps1
```

Both scripts stop unless you are logged in and the active subscription name contains "Azure for Students" (change it with `-ExpectedSubscription`).
The image is only set when the container app is first created: the Terraform file ignores later image changes, because the deploy workflow updates the image outside Terraform.
To update the image by hand: `az containerapp update --name <app> --resource-group <group> --image <registry>/complaint-classifier:<tag>`.

Steps, including how to create the OIDC credentials (not tested yet): [infra/README.md](infra/README.md). Check the costs in the Azure portal before you deploy. This README states no price.

Limits:
- It serves the TF-IDF model only, because the weights of the fine-tuned encoder were not saved.
- Scale to zero means a cold start: the first request after a quiet period has to wait for a container to start.
- The endpoint is public and has no login. So there is a size limit on the text (5,000 characters) and at most one replica.
- Student credit expires, and then the app stops working or costs real money.
- The deploy workflow and the Terraform apply were not run, and the OIDC steps are untested.

Deployed on: not yet

## Project layout

```
data/                data/README.md has the source, format, terms and every preparation step
src/prepare/         download, load, clean, labels, dedupe, splits, the guarded test access
src/baselines/       TF-IDF and embedding systems, tuning and final runs, model export for the API
src/finetune/        training script and the Colab/Kaggle notebook
src/llm/             zero-shot prompt, answer parser, cache, rate limit handling, providers (Groq, Z.ai), run script
src/evaluate/        metrics, confusion analysis, thresholds, results table, label-in-text check, paired test
api/                 FastAPI /classify with the abstain option, /health and the model version
deploy/model/        the exported model, its model card and its info file (what the Docker image contains)
scripts/             MLflow tracking and export of the registered model, test fixture builder
infra/               Terraform for Azure Container Apps, deploy and teardown scripts, OIDC steps
tests/               116 tests, a small real-data file (tests/fixtures), no GPU or key needed
docs/learn/          concept notes with walkthroughs, interview questions and glossaries
results/             results.md, label_in_text.md, paired_test.md, error analysis, predictions, tuning files, access log, Groq cache
```

## Limits and things to know

- **The data changes every day.** NHTSA updates its files daily, and the data of this repository was downloaded on 2026-10-20. A fresh download gives different splits, so results are only comparable on the saved prepared dataset
  (`data/processed/dataset.csv.gz`, not in git) and its fingerprints in `results/data_report.json`. Keep a copy of that file.
- **Label noise.** Scores are agreement with NHTSA's label. Two reasonable people can pick different components for the same defect.
- **Taxonomy and drift.** NHTSA has two similar engine components, and the class mix changes over the years. The time split shows how much this matters.
- **Sampled data.** The experiments use a random sample of 70,000 of the 554,344 unique single-component complaints, so that they run on a laptop. Results on all data could differ slightly.
- **`OTHER` is not one thing.** It mixes NHTSA's own catch-all and every component outside the 14 named classes, so it is hard to predict by design.
- **Complaints can name the component themselves.** Some texts include the recall's component name (for example "(Engine and Engine Cooling)"). I did not remove this, because it is part of the real data.
  The complaint's own label name in brackets is in the text for 8.8 percent of the random-split test complaints (1,223 of 13,840) and 6.7 percent of the time-split ones (932 of 13,840).
  TF-IDF scores 0.982 accuracy on those complaints and 0.746 on the rest of the random split, and 0.820 against 0.705 on the time split.
  The matching rule, a looser rule and the caveats are in [results/label_in_text.md](results/label_in_text.md); the difference does not prove that the model used the label name.
- **Zero-shot on this task.** The prompt contains only the class names, and a free-plan model was used (`glm-4.5-flash`, 5 of 1,000 answers were not a valid class; Groq `gpt-oss-20b`, 43 of 1,000). It cannot know NHTSA's labelling habits, such as when staff use `ENGINE` and when `ENGINE AND ENGINE COOLING`.
- **Latency is measured on one laptop CPU**, so absolute values will differ elsewhere. The Groq latency will include network time.
- **A discarded first run.** My first full run read the NHTSA files with the wrong text encoding (garbled quotes, for example "doesnât"). I found this while reading real examples, fixed it
  (the files are UTF-8), and ran everything again from scratch. No decision was based on the first run's results, and its test-access log is kept in `results/discarded_run1/` for transparency.
  A test now checks the decoding.

## Data source and licence

NHTSA vehicle owner complaints, https://www.nhtsa.gov/nhtsa-datasets-and-apis. NHTSA's Terms of Use say that information on the site is public information that may be distributed or copied, and that NHTSA
makes no guarantee of accuracy. Details and the exact wording are in [data/README.md](data/README.md). The raw data is not committed.

Code: MIT, see `LICENSE`.
