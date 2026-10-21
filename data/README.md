# Data: NHTSA vehicle owner complaints

## Source

The National Highway Traffic Safety Administration (NHTSA, part of the US Department of Transportation)
publishes the complaints that vehicle owners file about safety defects.

- Download page: https://www.nhtsa.gov/nhtsa-datasets-and-apis (section "Complaints")
- Files used here, the five-year sets by received date (checked on 2026-10-20):
  - https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2015-2019.zip (about 78 MB)
  - https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip (about 72 MB)
  - https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2025-2026.zip (about 40 MB)
- Layout description from NHTSA: https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt
- Also listed as dataset "NHTSA's Office of Defects Investigation (ODI) - Complaints" on data.transportation.gov.

`python -m src.prepare.download` downloads the three files into `data/raw/`. They are not committed:
they are large (about 190 MB zipped, 870 MB unzipped) and anybody can download them from the source.

## File format

Tab-separated text, 51 fields per row, **no header row**, dates as `YYYYMMDD`. We read these fields
(position in the file, name in `CMPL.txt`):

| Position | NHTSA name | Meaning | Used as |
| --- | --- | --- | --- |
| 1 | `ODINO` | NHTSA reference number of the complaint | complaint id |
| 12 | `COMPDESC` | component, for example `ENGINE AND ENGINE COOLING:ENGINE:GASOLINE` | the label |
| 17 | `LDATE` | date NHTSA received the complaint | time split |
| 20 | `CDESCR` | the complaint text written by the owner | the input text |
| 21 | `CMPL_TYPE` | source of the complaint (web form, hotline, letter, ...) | kept for information |
| 46 | `PROD_TYPE` | `V` vehicle, `T` tires, `E` equipment, `C` child seat | we keep `V` |

A complaint that concerns several components appears as **several rows with the same `ODINO`**, one
row per component. The component name has up to three levels separated by colons. Component names can
contain commas, for example `SERVICE BRAKES, HYDRAULIC`.

## Terms

NHTSA's Terms of Use (https://www.nhtsa.gov/about-nhtsa/terms-use) say: "Information presented on this web
site is considered public information and may be distributed or copied." The page also says that NHTSA makes no
claims or guarantees about the accuracy, completeness or adequacy of the contents. data.transportation.gov lists
the rights of the dataset as "public". The complaint data also contains personal data (city, state, partial VIN),
which is why only the text, label, id and date are kept in the prepared data.

What this repository commits from the data: a test file of 443 real complaint rows with only the fields above
(`tests/fixtures/raw_sample.tsv`), the counts of the preparation (`results/data_report.json`), the model predictions (ids and labels, no text),
the Groq answers (ids, raw answers and token counts, no text), and a few dozen shortened complaint texts that illustrate the biggest mistakes
(`results/errors_*.json`, `results/error_analysis.md`). The prepared dataset and the raw files stay on your computer.

## What the preparation does to the data

`python -m src.prepare.run` writes `data/processed/dataset.csv.gz` and the numbers below to `results/data_report.json`.

| Step | Rows or complaints left | Removed |
| --- | --- | --- |
| Raw rows, 2015 to October 2026 | 1,114,507 | |
| Vehicle complaints only (`PROD_TYPE` = V) | 1,101,224 | 13,283 other products |
| Complaints (`ODINO`) with at least 30 characters of text | 778,234 complaints | 10,070 rows with empty or very short text |
| **Multi-component complaints dropped** | 559,907 | **218,327 complaints (28.1%)** |
| Exact duplicate texts removed | 554,344 | 5,563 |
| Random sample of 70,000 (seed 42), so the experiments run on a laptop | 70,000 | |
| Near duplicates removed | **69,195** | 805 |

Notes on each step:

- **Labels.** The label is the top-level component, the part of `COMPDESC` before the first colon.
  The 14 most frequent top-level components keep their names. All others, including NHTSA's own
  catch-all `UNKNOWN OR OTHER`, become `OTHER`. That gives 15 classes.
- **Multi-component complaints** are dropped completely, because their label is not clear. A complaint that
  lists several sub-components of the same top-level component is kept once.
- **Near duplicates** are texts that share most of their word triples with an earlier text (cosine similarity of
  0.8 or more). They are removed before the data is split, so the same complaint can never be in both the training
  and the test set.
- **Splits.** `random`: 70 / 10 / 20 percent, stratified by class, seed 42. `time`: the oldest 70 percent of the
  complaints (2015-01-01 to 2023-04-19) for training, the next 10 percent for validation, and the newest 20 percent
  (2024-07-29 to 2026-10-18) for testing.

## Things to know about the labels

- **Label noise.** The component was assigned by NHTSA staff (hotline calls are typed by agents) or chosen by the owner
  on the complaint form. Two people can describe the same defect with different components. Accuracy in this project is
  agreement with that label, not with absolute truth.
- **The component list and its use change over time.** `ENGINE` and `ENGINE AND ENGINE COOLING` are two separate top-level
  components that cover similar problems, and `ENGINE AND ENGINE COOLING` is used much more in recent years (0.7 percent of the training
  part of the time split, 3.5 percent of its test part). `FORWARD COLLISION AVOIDANCE` hardly existed before 2020. Air bag complaints go the
  other way: 15.5 percent of the training part and only 3.5 percent of the test part. The class mix therefore changes a lot between the
  oldest and the newest complaints. This is the reason for the time split.
- **Sampling.** The experiments use a random sample of 70,000 of the 554,344 unique single-component complaints. Results on
  the full data could differ slightly.
