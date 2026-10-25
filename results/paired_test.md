# Fine-tuned encoder against TF-IDF on the same test items

Each test complaint is counted once, by which of the two systems predicted the NHTSA label. McNemar's exact two-sided test uses only the complaints where exactly one system is right.

| Split | Items | Both right | Both wrong | Only fine-tuned right | Only TF-IDF right | Exact two-sided p |
| --- | --- | --- | --- | --- | --- | --- |
| random | 13,840 | 9,973 | 2,081 | 1,144 | 642 | 7.8 x 10^-33 |
| time | 13,840 | 9,219 | 2,380 | 1,595 | 646 | 5.2 x 10^-92 |

The p value says whether the difference could be chance if the two systems were equally good on this test set. It does not say anything about other data, and it ignores that both systems were tuned on the same validation set.
