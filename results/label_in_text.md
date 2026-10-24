# Does the complaint text contain its own label name?

Matching rule (main): the text contains `(` + the complaint's own label name + `)`, ignoring case, for example `(ENGINE AND ENGINE COOLING)`. For the class OTHER, the NHTSA name `UNKNOWN OR OTHER` also counts. No fuzzy matching.

Why brackets: NHTSA staff write the recalled component in brackets after a campaign number, for example `NHTSA CAMPAIGN NUMBER: 14V355000 (ELECTRICAL SYSTEM)`. The bare words (`engine`, `steering`, `other`) appear in many complaints for ordinary reasons, so the looser rule in the second row is shown only for comparison.

What is ambiguous: the bracket text is the component of the recall, which is not always the complaint's label, and a complaint can name a different component in brackets. This file counts only a match with the complaint's own label. A match does not prove that the model used it. The accuracy difference also mixes in that these complaints are mostly about recalls, which may be easier or harder for other reasons.

Predictions: `tfidf_lr` on the complete test sets, from `results/predictions`.

## Random split (13,840 test complaints)

| Rule | Complaints with the label name | Share | TF-IDF accuracy on them | TF-IDF accuracy on the rest |
| --- | --- | --- | --- | --- |
| (LABEL) in brackets (main rule) | 1,223 | 8.8 percent | 0.982 | 0.746 |
| label words anywhere (looser) | 4,359 | 31.5 percent | 0.930 | 0.692 |

## Time split (13,840 test complaints)

| Rule | Complaints with the label name | Share | TF-IDF accuracy on them | TF-IDF accuracy on the rest |
| --- | --- | --- | --- | --- |
| (LABEL) in brackets (main rule) | 932 | 6.7 percent | 0.820 | 0.705 |
| label words anywhere (looser) | 4,415 | 31.9 percent | 0.877 | 0.636 |
