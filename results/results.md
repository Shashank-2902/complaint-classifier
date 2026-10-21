# Results

Accuracy is measured against the NHTSA component label, which was assigned by NHTSA staff or by the complaint form. That label has noise, so these numbers are agreement with the label, not with absolute truth.

## Random split

| Part | Rows | First date | Last date |
| --- | --- | --- | --- |
| train | 48435 | 2015-01-01 | 2026-10-18 |
| val | 6920 | 2015-01-02 | 2026-10-18 |
| test | 13840 | 2015-01-02 | 2026-10-18 |

### All test items

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 13840 | 0.767 | 0.742 | 2.7 | 0 (local) |
| Sentence embeddings + logistic regression | 13840 | 0.756 | 0.736 | 25.2 | 0 (local) |
| Fine-tuned small encoder | not run | not run | not run | not run | not run |
| Groq zero-shot | not run | not run | not run | not run | not run |

### The fixed subset of 1000 test items (the same items for every system)

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 1000 | 0.775 | 0.745 | 2.7 | 0 (local) |
| Sentence embeddings + logistic regression | 1000 | 0.769 | 0.727 | 25.2 | 0 (local) |
| Fine-tuned small encoder | not run | not run | not run | not run | not run |
| Groq zero-shot | not run | not run | not run | not run | not run |

### F1 per class (all test items)

| Class | TF-IDF + logistic regression | Sentence embeddings + logistic regression |
| --- | --- | --- |
| AIR BAGS | 0.94 | 0.95 |
| ELECTRICAL SYSTEM | 0.67 | 0.66 |
| ENGINE | 0.80 | 0.79 |
| ENGINE AND ENGINE COOLING | 0.64 | 0.64 |
| EXTERIOR LIGHTING | 0.87 | 0.90 |
| FORWARD COLLISION AVOIDANCE | 0.66 | 0.67 |
| FUEL/PROPULSION SYSTEM | 0.75 | 0.74 |
| POWER TRAIN | 0.80 | 0.78 |
| SERVICE BRAKES | 0.84 | 0.84 |
| STEERING | 0.93 | 0.92 |
| STRUCTURE | 0.64 | 0.63 |
| SUSPENSION | 0.73 | 0.72 |
| VEHICLE SPEED CONTROL | 0.52 | 0.48 |
| VISIBILITY/WIPER | 0.69 | 0.71 |
| OTHER | 0.63 | 0.60 |

### Confidence threshold (items below the threshold would go to a human)

| System | Threshold | Accuracy of accepted items | Coverage (share accepted) |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.559 | 0.882 | 0.703 |
| TF-IDF + logistic regression | 0.854 | 0.957 | 0.397 |
| Sentence embeddings + logistic regression | 0.627 | 0.870 | 0.699 |
| Sentence embeddings + logistic regression | 0.895 | 0.951 | 0.398 |

## Time split

| Part | Rows | First date | Last date |
| --- | --- | --- | --- |
| train | 48436 | 2015-01-01 | 2023-04-19 |
| val | 6919 | 2023-04-19 | 2024-07-29 |
| test | 13840 | 2024-07-29 | 2026-10-18 |

### All test items

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 13840 | 0.713 | 0.674 | 2.2 | 0 (local) |
| Sentence embeddings + logistic regression | 13840 | 0.705 | 0.662 | 23.4 | 0 (local) |
| Fine-tuned small encoder | not run | not run | not run | not run | not run |
| Groq zero-shot | not run | not run | not run | not run | not run |

### The fixed subset of 1000 test items (the same items for every system)

| System | Items | Accuracy | Macro-F1 | Latency per item (ms) | Cost per 1000 items |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 1000 | 0.735 | 0.678 | 2.2 | 0 (local) |
| Sentence embeddings + logistic regression | 1000 | 0.715 | 0.654 | 23.4 | 0 (local) |
| Fine-tuned small encoder | not run | not run | not run | not run | not run |
| Groq zero-shot | not run | not run | not run | not run | not run |

### F1 per class (all test items)

| Class | TF-IDF + logistic regression | Sentence embeddings + logistic regression |
| --- | --- | --- |
| AIR BAGS | 0.79 | 0.83 |
| ELECTRICAL SYSTEM | 0.57 | 0.61 |
| ENGINE | 0.78 | 0.79 |
| ENGINE AND ENGINE COOLING | 0.29 | 0.34 |
| EXTERIOR LIGHTING | 0.85 | 0.88 |
| FORWARD COLLISION AVOIDANCE | 0.59 | 0.55 |
| FUEL/PROPULSION SYSTEM | 0.67 | 0.70 |
| POWER TRAIN | 0.81 | 0.80 |
| SERVICE BRAKES | 0.80 | 0.78 |
| STEERING | 0.92 | 0.91 |
| STRUCTURE | 0.64 | 0.50 |
| SUSPENSION | 0.75 | 0.72 |
| VEHICLE SPEED CONTROL | 0.45 | 0.31 |
| VISIBILITY/WIPER | 0.57 | 0.60 |
| OTHER | 0.64 | 0.61 |

### Confidence threshold (items below the threshold would go to a human)

| System | Threshold | Accuracy of accepted items | Coverage (share accepted) |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.718 | 0.834 | 0.679 |
| TF-IDF + logistic regression | 0.949 | 0.920 | 0.386 |
| Sentence embeddings + logistic regression | 0.613 | 0.814 | 0.690 |
| Sentence embeddings + logistic regression | 0.875 | 0.910 | 0.378 |
