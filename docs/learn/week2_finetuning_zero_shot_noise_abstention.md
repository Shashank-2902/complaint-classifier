# Week 2: fine-tuning, zero-shot prompting, label noise and abstention

## 1. Concept

### What fine-tuning changes

A **pretrained language model** (here DistilBERT) has already read a huge amount of text and knows a lot about language: which
words go together, what "brakes" and "stopping" have in common. It does not yet know our task.

**Fine-tuning** continues the training on our labelled complaints. The model keeps what it knows about language and, in a few
epochs (passes over the data), learns to map complaint texts to the 15 classes. Compared with TF-IDF, it can use word order and
context ("the engine did not start" and "I started the engine" are different), and it handles words it has not seen.

It needs a GPU to train in a reasonable time. We use a free one (Google Colab or Kaggle) and keep the training in a script plus a notebook.

### Why an encoder suits classification

A language model can be built in two main shapes:

- an **encoder** (BERT, DistilBERT, DeBERTa) reads the whole text at once and produces a representation of it. A small classification
  layer on top picks the label. It is small (66 million parameters for DistilBERT), fast and cheap.
- a **generative model** (GPT-style, Llama, Qwen) writes text one word after another. It is much bigger (billions of parameters).

For "short text in, one of 15 labels out", an encoder is the natural tool: cheaper, faster, and usually as accurate as a fine-tuned
generative model. The question "why fine-tune a generative LLM for this?" has no good answer, which is why this project does not do it.
(A LoRA fine-tune of a small generative model is an optional extra row, only if time remains.)

### Zero-shot prompting and its limits

**Zero-shot** means asking a big model to do the task with no training examples. We put the class list and the complaint in the prompt and
ask for one label (temperature 0, so the answer is as repeatable as possible). It needs no training data and no GPU, and it works on day one.

Its limits show up in this project:

- **It does not know our label rules.** NHTSA has both `ENGINE` and `ENGINE AND ENGINE COOLING`. A model reading only the names cannot know which
  one NHTSA staff used for which kind of complaint. Training data teaches this, a prompt does not.
- **`OTHER` is vague.** The model sees "none of the above"; the real `OTHER` is a mix of many small components.
- **Cost and speed.** Every item is an API call: it takes hundreds of milliseconds and costs money per token. The classical models run in milliseconds and cost nothing per item.
- **Rate limits.** Free plans limit requests per minute and per day, so we run it on a fixed subset of about 1,000 test items.
- **Format errors.** The model may answer with a sentence or a name that is not in the list. We count these as invalid and report how many.

We cache every raw answer on disk, so a run can be repeated or resumed without new API calls and the answers can be inspected.

### Label noise

The label is whatever the NHTSA staff or the car owner picked. Different people describe the same defect with different components.
So some of the model's "mistakes" are cases where the label is arguable or wrong. This sets a ceiling: even a perfect model cannot reach 100 percent.
In the error analysis we read real misclassified texts and judge each case: model wrong, label wrong, or both defensible. Those judgements are
my reading of a few examples, not a measurement.

### Confidence and abstention

A classifier can say how sure it is. For logistic regression this is the probability of the predicted class; for the fine-tuned encoder it is the softmax probability.
**Abstention** means: if the confidence is below a **threshold**, do not answer automatically, send the item to a human ("needs human review").

There is a trade-off:

- a higher threshold: fewer items are answered automatically (lower **coverage**), but the answered ones are more often correct;
- a lower threshold: more coverage, lower accuracy on the answered items.

We choose two thresholds on the **validation** set (the ones that would accept about 70 percent and 40 percent of validation items), and then report the accuracy
and coverage on the test set for each. The API uses the same idea: the response contains `needs_human_review`.

Note: confidence from a model is not always honest. A model can be confidently wrong, and the numbers can change on new kinds of data (time split), so the threshold should be re-checked regularly.

## 2. Code walkthrough

**`src/finetune/`**

- `data.py`: helpers without any GPU code (label to number maps, batches, softmax).
- `train.py`: loads DistilBERT, trains for 3 epochs on the training set, scores validation macro-F1 after every epoch, keeps the best epoch, opens the test set once, and writes
  predictions with confidence and the measured latency. `--smoke` runs a tiny version on validation rows to check the code, never on the test set.
- `colab_finetune.ipynb`: the notebook for Colab or Kaggle. It prepares the data again with the same seed and **checks that the splits equal the repository's** before training.

**`src/llm/`**

- `prompt.py`: the prompt with the class list, and the parser that turns the answer into a class or `INVALID`.
- `cache.py`: one JSON line per answer (key, id, raw answer, token counts, latency). The complaint text is not stored.
- `groq_zero_shot.py`: sends the items, spaces the calls to stay under the rate limit, waits and retries on HTTP 429, stops cleanly on a daily limit and computes the cost from token counts.
- `run_groq.py`: runs the fixed subset and writes predictions in the same format as every other system.

**`src/evaluate/`**: `report.py` builds the results tables from the prediction files and writes "not run" for any missing system. `error_analysis.py` makes confusion matrices and the five most frequent mistakes with real texts. `subsets.py` creates the fixed 1,000-item subset.

**`api/main.py`**: `POST /classify` returns the label, the confidence, the top three classes and `needs_human_review`. It serves the TF-IDF model.

**Tests** use a fake model for Groq and a fake encoder, so they need no key and no download. They check, among other things: the answer parser, the cache, rate limiting, retries, that a table never contains an invented number, and the abstain behaviour of the API.

## 3. Interview questions

1. **What does fine-tuning change in a pretrained model?**
   It adapts the model's weights to the task by continuing training on labelled examples, so it keeps its language knowledge and learns the label mapping.

2. **Why an encoder like DistilBERT and not a big generative model?**
   For short-text classification an encoder is cheaper, faster and usually as accurate. A generative model adds cost without a clear benefit.

3. **What is zero-shot prompting and where does it struggle here?**
   Asking a model to classify with only the class names. It cannot know NHTSA's own labelling habits, such as when staff use `ENGINE` and when `ENGINE AND ENGINE COOLING`.

4. **Why is the Groq row run on a subset of 1,000 items?**
   Rate limits and cost. All other systems are also reported on the same 1,000 items, so the comparison is fair.

5. **Why temperature 0 and cached outputs?**
   Temperature 0 makes answers as repeatable as possible. The cache means a run can be resumed or repeated without new calls, and raw answers can be inspected.

6. **How did you pick hyperparameters without touching the test set?**
   On the validation set only. The test set is opened once per system and split, and every access is logged.

7. **What is label noise and how does it affect the results?**
   Labels that are wrong or arguable. Scores measure agreement with NHTSA's label, so some errors are the label's fault. Perfect accuracy is not reachable.

8. **What is a confidence threshold and what trade-off does it create?**
   Below the threshold the item goes to a human. A higher threshold gives higher accuracy on the answered items but lower coverage.

9. **How did you choose the thresholds?**
   On the validation set, so that about 70 percent and 40 percent of validation items are accepted. Then the test accuracy and coverage at those fixed thresholds are reported.

10. **Why did you not fine-tune a generative LLM?**
    A small encoder does the same job for a fraction of the cost. A LoRA run is an optional extra row, not part of the core result.

## 4. Glossary

- **Pretrained model**: a model trained on a large amount of general text.
- **Fine-tuning**: continuing the training on your own labelled data.
- **Epoch**: one pass over the training data.
- **Encoder / generative model**: a model that builds a representation of a text / one that writes text.
- **Zero-shot**: solving a task with no training examples, only an instruction.
- **Prompt**: the text sent to a language model.
- **Temperature**: randomness of a model's output; 0 means as repeatable as possible.
- **Rate limit**: the number of requests an API allows per minute or per day.
- **Cache**: saved answers, so they do not have to be requested again.
- **Label noise**: wrong or arguable labels.
- **Confidence**: the probability a model gives to its chosen class.
- **Threshold**: the confidence level below which a human decides.
- **Abstention**: the model declines to answer.
- **Coverage**: the share of items the model answers on its own.
- **Latency**: the time to classify one item.
