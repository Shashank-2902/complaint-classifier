# Week 1: TF-IDF, baselines, data leakage, random versus time split, macro-F1

## 1. Concept

### The task

A car owner writes a short text about a problem. NHTSA stores the text together with the component the problem is about
(engine, air bags, steering, and so on). We train models that read the text and predict the component. This is **text
classification**: one text in, one label out, out of a fixed list of 15 classes (14 components plus `OTHER`).

### What TF-IDF does

A computer cannot read words, so we turn every text into a list of numbers. **TF-IDF** does this in a simple way:

- **TF** (term frequency): how often a word appears in this text.
- **IDF** (inverse document frequency): how rare the word is across all texts. A word that is in almost every text (like
  "the") gets a low weight. A word that is in few texts (like "airbag") gets a high weight.

Every text becomes a long list with one number per word (and per word pair). The numbers are high for words that are typical
for this text. **Logistic regression** is then trained on these lists. For every class it learns which words push towards that
class. It is fast, easy to inspect and often surprisingly hard to beat.

### Why a baseline matters

A **baseline** is a simple method you try first. Without it you cannot know if a complicated model is worth the effort.
If a language model gets 70 percent and TF-IDF gets 72 percent, the language model was not worth its cost. A baseline can
win, and reporting that honestly is part of the project. Without a baseline, a number like "70 percent accuracy" means
nothing.

### Sentence embeddings (the middle option)

A **sentence-transformer** reads a whole text and returns a vector (here 384 numbers) that captures its meaning. Texts with
similar meaning have similar vectors, even when they use different words ("brakes failed" and "could not stop the car").
We train logistic regression on these vectors. It needs no GPU, only some patience for the embedding step (done once and
saved on disk).

### Data leakage

**Leakage** means information from the test set sneaks into training. Then the test score is too good, and it will not hold
up in real use. In this project there are three leaks to avoid:

1. **Duplicates.** NHTSA data contains the same complaint several times (the same text filed for several vehicles, or copied with
   small changes). If one copy is in training and another in the test set, the model only has to remember it. We remove exact and
   near duplicates **before** splitting.
2. **Tuning on the test set.** If you try ten settings and keep the one that scores best on the test set, you have used the test
   set to train. We pick settings on a separate **validation set**. The test set is only used for the final table. The code
   refuses a second normal use of the same system and split, and every access is written to `results/test_access_log.jsonl`. A resumed run is let through and marked
   `forced = true`. The classical systems and the Z.ai run each have one access per split. The Groq run was resumed six times over several days (rate limits) and each resume
   read the same 1,000 items again; the README lists the log. The log does not see Colab sessions: each starts from a fresh clone with an empty log.
3. **Same complaint, many rows.** A complaint with several components appears in several rows with the same id. We drop these
   complaints, so no complaint can be in two parts.

### Random split versus time split

- **Random split.** Every complaint is assigned randomly to train, validation or test, keeping the class shares equal. The test
  complaints look just like the training ones. This is the easy case, and it tends to give optimistic numbers.
- **Time split.** The oldest 70 percent of the complaints are for training, the newest 20 percent for testing. This is how the
  model would really be used: it is trained on the past and must classify new complaints. If the world changes, the score drops.
  Here it does change: new technology (forward collision avoidance) appeared, the second engine component
  (`ENGINE AND ENGINE COOLING`) is used much more now, and air bag complaints were 15.5 percent of the training data but only 3.5 percent of the newest data.

Reporting both shows how much of the performance is real and how much comes from an easy test.

### Why macro-F1 next to accuracy

The classes are imbalanced: `OTHER` is 19 percent of the data, `FORWARD COLLISION AVOIDANCE` only 1 percent. **Accuracy** counts
every complaint equally, so a model can reach a decent score by ignoring small classes. **F1** for one class combines precision
(how many of its predictions were right) and recall (how many of its real items it found). **Macro-F1** is the average of the F1
scores of all classes, so every class counts the same, no matter how small. A big gap between accuracy and macro-F1 means the model is weak on small classes.

## 2. Code walkthrough

**`src/prepare/`** builds the dataset from the raw files.

- `load_raw.py` reads the NHTSA zip files (tab separated, no header).
- `labels.py` cuts the component at the first colon, drops complaints with several top-level components and keeps the 14 most
  frequent classes plus `OTHER`.
- `clean.py` fixes whitespace and drops very short texts.
- `dedupe.py` removes exact duplicates and near duplicates (texts that share most of their word triples).
- `splits.py` makes the random and the time split.
- `dataset.py` loads the data. Tuning code gets only train and validation. The test set comes from `load_test_once`, which refuses a second use.
- `run.py` runs all steps and writes the counts of every step to `results/data_report.json`.

**`src/baselines/`** has the two classical systems.

- `tfidf.py`: TF-IDF (1 and 2 word groups, 100,000 features) and logistic regression. It tries five values of C and keeps the best on validation macro-F1.
- `embeddings.py`: MiniLM embeddings (cached on disk) and logistic regression.
- `run_baseline.py`: stage `tune` picks C and two confidence thresholds on the validation set; stage `final` trains with that C and predicts the test set once.

**`src/evaluate/`** has the metrics (accuracy, macro-F1, F1 per class), the confusion matrix and the table writer. A system without a prediction file is shown as "not run".

**Tests** check, among other things: the multi-component rule, that duplicates are removed and no text is in two parts, that the time split puts older complaints in training, that the split is repeatable, that tuning cannot reach the test set, and that the test set can be opened only once.

## 3. Interview questions

1. **What does TF-IDF do?**
   It turns a text into numbers. Each word gets a weight that is high if the word is frequent in this text and rare in the others.

2. **Why start with a TF-IDF baseline?**
   It is quick, strong and a fair reference. If a bigger model does not clearly beat it, the bigger model is not worth its cost.

3. **What is data leakage? Give an example from this project.**
   Information from the test set getting into training. Example: the same complaint text in both the training and the test set,
   which is why near duplicates are removed before splitting.

4. **Why do you need a validation set if you have a test set?**
   Settings must be chosen on data that is not the test set. Otherwise the test score is optimistic. The test set is only used at the end, and every access is logged.

5. **How do you make sure the test set is only used once?**
   Tuning code only receives train and validation data. The test set is loaded through one function that logs the access and refuses a second use. A test checks this.

6. **What is the difference between a random split and a time split?**
   Random: test complaints look like training ones. Time: train on old complaints and test on new ones, which shows how the model handles change.

7. **Why did you remove near duplicates and not only exact ones?**
   Copies with small edits (an added sentence, a changed word) would pass an exact check but still let the model remember instead of learn.

8. **Why report macro-F1 and not only accuracy?**
   The classes are imbalanced. Accuracy can look good while small classes are ignored. Macro-F1 gives every class the same weight.

9. **Why did you drop complaints with several components?**
   Their label is not clear: one text, several correct answers. Keeping them would make the task ill-defined. We report how many were dropped (28 percent of the complaints).

10. **What does the label noise do to your results?**
    The label is what NHTSA staff or the owner chose. Two reasonable people can pick different components for the same defect, so some
    "errors" are the label's fault. Scores are agreement with the label, not with the truth.

## 4. Glossary

- **Text classification**: predicting a label from a text.
- **TF-IDF**: word weights based on how often a word is in this text and how rare it is overall.
- **Logistic regression**: a simple model that learns a weight for every feature and class.
- **Baseline**: a simple method used as a reference.
- **Embedding**: a vector of numbers that represents the meaning of a text.
- **Leakage**: test information that reaches training and inflates the score.
- **Near duplicate**: a text that is almost the same as another one.
- **Validation set**: data used to choose settings.
- **Test set**: data used once for the final score.
- **Random split / time split**: dividing data randomly / by date.
- **Accuracy**: share of correct predictions.
- **Precision / recall / F1**: how many predictions are right / how many real items are found / their balance.
- **Macro-F1**: the average F1 over all classes.
- **Class imbalance**: some classes are much more frequent than others.
- **Label noise**: labels that are wrong or arguable.
