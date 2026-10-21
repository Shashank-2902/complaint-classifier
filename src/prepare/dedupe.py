"""Remove duplicate complaints before splitting.

If the same complaint (or a copy with small edits) were in both the training and the test set,
the test score would be too good. So duplicates are removed from the whole data first.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from src import config
from src.prepare.clean import normalized_key


def sort_oldest_first(frame):
    return frame.sort_values(["received", "odi_no"], kind="mergesort")


def drop_exact_duplicates(frame):
    """Keep the oldest copy of every identical text. Returns (frame, number removed)."""
    ordered = sort_oldest_first(frame)
    duplicated = ordered["text"].map(normalized_key).duplicated(keep="first")
    return ordered[~duplicated], int(duplicated.sum())


def find_near_duplicates(texts, threshold=config.NEAR_DUPLICATE_SIMILARITY, chunk=2000):
    """Return the positions of texts that are near copies of an EARLIER text in the list.

    Texts are compared by the word triples (three words in a row) they share, using cosine
    similarity. Two unrelated texts share almost no triples, but a copy with small edits
    shares most of them. The list must be sorted oldest first, so the oldest copy stays.
    """
    # Ignore word triples that appear in more than 1 percent of the texts: they are boilerplate, not a sign of copying.
    max_df = max(10, int(0.01 * len(texts)))
    counter = CountVectorizer(ngram_range=(3, 3), binary=True, min_df=2, max_df=max_df, lowercase=True)
    try:
        vectors = normalize(counter.fit_transform(texts)).tocsr()
    except ValueError:      # no word triple is shared by two texts, so there can be no copies
        return np.array([], dtype=int)
    duplicates = set()

    for start in range(0, vectors.shape[0], chunk):
        block = vectors[start:start + chunk]
        similarity = (block @ vectors.T).tocoo()
        for row, column, value in zip(similarity.row, similarity.col, similarity.data):
            if value >= threshold and column < start + row:   # column is an earlier text
                duplicates.add(start + row)
    return np.array(sorted(duplicates), dtype=int)


def drop_near_duplicates(frame, threshold=config.NEAR_DUPLICATE_SIMILARITY):
    """Remove near copies, keeping the oldest. Returns (frame, number removed)."""
    ordered = sort_oldest_first(frame)
    positions = find_near_duplicates(ordered["text"].tolist(), threshold)
    keep = np.ones(len(ordered), dtype=bool)
    keep[positions] = False
    return ordered[keep], int(len(positions))
