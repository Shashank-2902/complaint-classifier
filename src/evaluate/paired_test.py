"""Fine-tuned encoder against TF-IDF on the same test items, with an exact McNemar test.

    python -m src.evaluate.paired_test

For every test item we look at which of the two systems got it right:
    both right, both wrong, only the fine-tuned encoder right (b), only TF-IDF right (c).
The items where both agree on right or wrong say nothing about which system is better. McNemar's test
looks only at b and c. If the two systems were equally good, b and c would be about equal: each
disagreement would go to either side with probability one half. The exact two-sided p value is
    p = 2 * sum(C(n, i) for i = 0 .. min(b, c)) / 2**n   with n = b + c, at most 1.
It is computed with integers (math.comb), no scipy.

This only reads results/predictions and data/processed. It does not open the test set through the guarded
function (that would be logged), it reads the labels the same way src/evaluate/report.py does.
"""
from math import comb, log10

from src import config
from src.evaluate.predictions import read_predictions
from src.prepare import dataset

FIRST, SECOND = "finetuned_encoder", "tfidf_lr"
OUTPUT = config.RESULTS_DIR / "paired_test.md"


def mcnemar_counts(truth, pred_first, pred_second):
    """(both right, both wrong, only first right, only second right) for lists of the same length."""
    both_right = both_wrong = only_first = only_second = 0
    for label, first, second in zip(truth, pred_first, pred_second):
        first_ok, second_ok = first == label, second == label
        if first_ok and second_ok:
            both_right += 1
        elif not first_ok and not second_ok:
            both_wrong += 1
        elif first_ok:
            only_first += 1
        else:
            only_second += 1
    return both_right, both_wrong, only_first, only_second


def mcnemar_exact(b, c):
    """The exact two-sided McNemar p value as (numerator, denominator) integers, so that p = numerator / denominator."""
    n = b + c
    if n == 0:
        return 1, 1
    numerator = 2 * sum(comb(n, i) for i in range(min(b, c) + 1))
    denominator = 2 ** n
    if numerator >= denominator:
        return 1, 1
    return numerator, denominator


def p_text(numerator, denominator):
    """The p value as text. Very small values are written as a power of ten, because a float would underflow."""
    if numerator >= denominator:
        return "1.000"
    log_p = log10(numerator) - log10(denominator)
    if log_p > -4:
        return f"{numerator / denominator:.4f}"
    exponent = int(log_p // 1)
    return f"{10 ** (log_p - exponent):.1f} x 10^{exponent}"


def compare(scheme):
    first, _ = read_predictions(FIRST, scheme)
    second, _ = read_predictions(SECOND, scheme)
    if first is None or second is None:
        raise SystemExit(f"missing predictions for the {scheme} split")
    truth = dataset.load_all()[["odi_no", "label"]]
    merged = first.merge(second, on="odi_no", suffixes=("_first", "_second")).merge(truth, on="odi_no")
    counts = mcnemar_counts(merged["label"], merged["pred_first"], merged["pred_second"])
    numerator, denominator = mcnemar_exact(counts[2], counts[3])
    return len(merged), counts, p_text(numerator, denominator)


def main():
    lines = ["# Fine-tuned encoder against TF-IDF on the same test items", "",
             "Each test complaint is counted once, by which of the two systems predicted the NHTSA label. "
             "McNemar's exact two-sided test uses only the complaints where exactly one system is right.", "",
             "| Split | Items | Both right | Both wrong | Only fine-tuned right | Only TF-IDF right | Exact two-sided p |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for scheme in config.SPLIT_SCHEMES:
        n, (both_right, both_wrong, only_first, only_second), p = compare(scheme)
        lines.append(f"| {scheme} | {n:,} | {both_right:,} | {both_wrong:,} | {only_first:,} | {only_second:,} | {p} |")
    lines += ["", "The p value says whether the difference could be chance if the two systems were equally good on this test set. "
              "It does not say anything about other data, and it ignores that both systems were tuned on the same validation set.", ""]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
