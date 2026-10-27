# Model card: complaint-classifier-tfidf

TF-IDF (1 to 2 word n-grams) and logistic regression that predicts the vehicle component of an NHTSA complaint text.

| Item | Value |
| --- | --- |
| Registered model | complaint-classifier-tfidf |
| Version | 1 |
| MLflow run | 8c23a8e1b1c748e9b00d6c72b68d7274 |
| Exported (UTC) | 2026-10-26T16:57:26+00:00 |
| Split used | time split (train and validation parts, 55355 complaints) |
| Regularisation C | 30 |
| Confidence thresholds | [0.7179729826168475, 0.94876346074956] (below the first one the API asks for human review) |
| Git commit of the training run | 9b6dbeaff0769300a82a0aff7a0c963a1bac1ca9 (working tree changed: True) |

## Validation metrics

Scored on the validation part with a model fitted on the training part only. The registered model above was then refitted on training plus validation, so it has no
honest validation score of its own.

| Metric | Value |
| --- | --- |
| Validation accuracy | 0.7303 |
| Validation macro-F1 | 0.6928 |

## Published test metrics

Copied from `results/results.md`, not computed by this export. Split: time, 13840 test complaints.

| Metric | Value |
| --- | --- |
| Test accuracy | 0.713 |
| Test macro-F1 | 0.674 |

## Data fingerprints

A short code for the list of complaint ids of each split, from `results/data_report.json`. The same fingerprints mean the same split.

| Split | Fingerprint |
| --- | --- |
| random_test | `61fdf66347a331be` |
| random_train | `ed4f99d737e105b4` |
| random_val | `8e1aa3884a9ff111` |
| time_test | `c6d84548fafe1d62` |
| time_train | `b9f3550c176f6d69` |
| time_val | `6ddc9383b4ca9996` |

## Limits

- Public US data (NHTSA vehicle owner complaints). Labels were assigned by NHTSA and have noise.
- One training run, no search over settings beyond the choice of C on the validation set.
- The fine-tuned encoder scores higher in this project, but its weights were not saved, so this classical model is the one that is served.
- The export did not open any test set.
