"""Two ways to split the data into train, validation and test.

random   a random split that keeps the class shares the same in every part
time     the oldest complaints are used for training, the newest for testing

The random split tests the model on easy cases (the test complaints look like the training ones).
The time split shows how the model holds up on newer complaints, which is what happens in real use.
"""
from sklearn.model_selection import train_test_split

from src import config


def random_split(frame, seed=config.SEED):
    """Return a Series with 'train', 'val' or 'test' for every row, stratified by label."""
    test_share = 1 - config.TRAIN_SHARE - config.VAL_SHARE
    train_val, test = train_test_split(frame.index, test_size=test_share, random_state=seed, stratify=frame["label"])
    val_share_of_rest = config.VAL_SHARE / (config.TRAIN_SHARE + config.VAL_SHARE)
    train, val = train_test_split(train_val, test_size=val_share_of_rest, random_state=seed,
                                  stratify=frame.loc[train_val, "label"])
    parts = frame["label"].copy().astype(object)
    parts.loc[train], parts.loc[val], parts.loc[test] = "train", "val", "test"
    return parts


def time_split(frame):
    """Oldest 70 percent train, the next 10 percent validation, the newest 20 percent test."""
    ordered = frame.sort_values(["received", "odi_no"], kind="mergesort")
    n = len(ordered)
    n_train, n_val = int(n * config.TRAIN_SHARE), int(n * config.VAL_SHARE)
    parts = frame["label"].copy().astype(object)
    parts.loc[ordered.index[:n_train]] = "train"
    parts.loc[ordered.index[n_train:n_train + n_val]] = "val"
    parts.loc[ordered.index[n_train + n_val:]] = "test"
    return parts


def add_splits(frame):
    """Add one column per scheme: split_random and split_time."""
    frame = frame.copy()
    frame["split_random"] = random_split(frame)
    frame["split_time"] = time_split(frame)
    return frame
