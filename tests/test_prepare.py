import pandas as pd

from src import config
from src.prepare import clean, dedupe, labels, splits


# ---------- labels ----------

def test_top_level_component_cuts_at_the_first_colon():
    assert labels.top_level_component("ENGINE AND ENGINE COOLING:ENGINE:GASOLINE") == "ENGINE AND ENGINE COOLING"
    assert labels.top_level_component("air bags:frontal") == "AIR BAGS"


def test_commas_inside_a_component_name_are_kept():
    assert labels.top_level_component("SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS") == "SERVICE BRAKES, HYDRAULIC"


def test_missing_component_gives_empty_label():
    assert labels.top_level_component(None) == ""


def make_rows(*rows):
    return pd.DataFrame(rows, columns=["odi_no", "top_level"])


def test_multi_component_complaints_are_dropped_and_counted():
    frame = make_rows(("1", "ENGINE"), ("2", "ENGINE"), ("2", "AIR BAGS"), ("3", "STEERING"), ("3", "STEERING"))
    kept, dropped = labels.drop_multi_component(frame)
    assert dropped == 1                                   # complaint 2 lists two components
    assert sorted(kept["odi_no"]) == ["1", "3"]           # complaint 3 only repeats one component, so it stays once


def test_most_frequent_classes_ignore_the_nhtsa_catch_all():
    frame = make_rows(*[(str(i), "ENGINE") for i in range(5)], *[(str(i), "UNKNOWN OR OTHER") for i in range(10, 30)],
                      *[(str(i), "AIR BAGS") for i in range(40, 43)], ("99", "WHEELS"))
    assert labels.most_frequent_classes(frame, top_k=2) == ["ENGINE", "AIR BAGS"]


def test_rare_and_catch_all_components_become_other():
    frame = make_rows(("1", "ENGINE"), ("2", "WHEELS"), ("3", "UNKNOWN OR OTHER"))
    mapped = labels.map_labels(frame, ["ENGINE"])
    assert mapped["label"].tolist() == ["ENGINE", config.OTHER, config.OTHER]


# ---------- cleaning ----------

def test_clean_text_collapses_whitespace_and_unescapes():
    assert clean.clean_text("  the   car\n\nstalled &amp; died \t") == "the car stalled & died"
    assert clean.clean_text(None) == ""


def test_short_texts_are_dropped():
    frame = pd.DataFrame({"text": ["too short", "this complaint text is long enough to keep for training"]})
    kept, dropped = clean.clean_frame(frame)
    assert len(kept) == 1 and dropped == 1


# ---------- duplicates ----------

def test_exact_duplicates_keep_the_oldest_copy():
    frame = pd.DataFrame({"odi_no": ["2", "1", "3"], "received": pd.to_datetime(["2020-02-01", "2020-01-01", "2020-03-01"]),
                          "text": ["The Brakes failed.", "the   brakes failed.", "something else entirely"]})
    kept, removed = dedupe.drop_exact_duplicates(frame)
    assert removed == 1
    assert "1" in kept["odi_no"].tolist() and "2" not in kept["odi_no"].tolist()


LONG = ("while driving on the highway at about sixty miles per hour the steering wheel suddenly locked and the dashboard "
        "showed a warning light the dealer replaced the power steering pump but the problem returned two weeks later")
OTHER_TEXT = ("the transmission slipped between second and third gear and the engine revved without the car moving "
              "the shop said the clutch pack was worn and quoted a very high price for the repair")


def test_near_duplicates_are_found_and_the_oldest_stays():
    texts = [LONG, OTHER_TEXT, LONG + " please help", LONG.replace("sixty", "fifty"), OTHER_TEXT + " thanks"]
    duplicates = dedupe.find_near_duplicates(texts, threshold=0.7)
    assert set(duplicates.tolist()) == {2, 3, 4}          # copies of text 0 and text 1, never the first copy


def test_unrelated_texts_are_not_near_duplicates():
    assert len(dedupe.find_near_duplicates([LONG, OTHER_TEXT], threshold=0.7)) == 0


# ---------- the whole preparation on the real-data fixture ----------

def test_build_report_adds_up(built):
    _, report = built
    assert report["raw_rows"] == 443
    assert report["vehicle_rows"] == 439                       # 4 rows are tires, equipment or child seats
    assert report["multi_component_complaints_dropped"] == 12
    assert report["exact_duplicates_removed"] >= 10
    assert report["near_duplicates_removed"] >= 5
    assert report["final_complaints"] == len(built[0])


def test_every_complaint_is_in_the_dataset_once_with_enough_text(built):
    frame, _ = built
    assert frame["odi_no"].is_unique
    assert (frame["text"].str.len() >= config.MIN_TEXT_CHARS).all()
    assert not frame["text"].map(clean.normalized_key).duplicated().any()


def test_text_is_decoded_as_utf8_without_garbled_characters(raw_sample, built):
    """NHTSA files are UTF-8. Reading them as latin-1 turns curly quotes into 'â' sequences."""
    texts = raw_sample["text"]
    assert not texts.str.contains("â|Ã", regex=True).any()
    assert texts.str.contains("[‘’“”]", regex=True).any()      # real curly quotes are still there
    assert not built[0]["text"].str.contains("â|Ã", regex=True).any()


def test_labels_are_the_frequent_classes_plus_other(built):
    frame, report = built
    assert set(frame["label"]) <= set(report["classes"]) | {config.OTHER}
    assert config.OTHER in set(frame["label"])


# ---------- splits ----------

def test_split_sizes_are_about_70_10_20(built):
    frame, _ = built
    for column in ["split_random", "split_time"]:
        shares = frame[column].value_counts(normalize=True)
        assert abs(shares["train"] - 0.70) < 0.03
        assert abs(shares["val"] - 0.10) < 0.03
        assert abs(shares["test"] - 0.20) < 0.03


def test_random_split_keeps_every_class_in_every_part(built):
    frame, _ = built
    for part in ["train", "val", "test"]:
        assert set(frame.loc[frame["split_random"] == part, "label"]) == set(frame["label"])


def test_time_split_trains_on_older_complaints_than_it_tests_on(built):
    frame, _ = built
    by_part = {part: frame[frame["split_time"] == part]["received"] for part in ["train", "val", "test"]}
    assert by_part["train"].max() <= by_part["val"].min()
    assert by_part["val"].max() <= by_part["test"].min()


def test_the_same_text_is_never_in_two_parts(built):
    """After removing duplicates, no complaint can be in the training and the test set at the same time."""
    frame, _ = built
    for column in ["split_random", "split_time"]:
        keys = frame["text"].map(clean.normalized_key)
        parts_per_text = frame.assign(key=keys).groupby("key")[column].nunique()
        assert (parts_per_text == 1).all()


def test_splits_are_repeatable(built):
    frame, _ = built
    again = splits.add_splits(frame.drop(columns=["split_random", "split_time"]))
    assert again["split_random"].tolist() == frame["split_random"].tolist()
    assert again["split_time"].tolist() == frame["split_time"].tolist()
