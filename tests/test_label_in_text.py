import pandas as pd

from src.evaluate import label_in_text


def test_the_label_in_brackets_is_found_ignoring_case():
    text = "THE VIN WAS NOT INCLUDED IN NHTSA CAMPAIGN NUMBER: 14V153000 (steering). THE MILEAGE WAS 10"
    assert label_in_text.bracket_match(text, "STEERING")
    assert label_in_text.bracket_match("(ENGINE AND ENGINE COOLING) failed", "ENGINE AND ENGINE COOLING")


def test_the_bare_word_or_another_label_does_not_count_as_a_bracket_match():
    assert not label_in_text.bracket_match("THE STEERING WHEEL LOCKED", "STEERING")
    assert not label_in_text.bracket_match("CAMPAIGN 14V153000 (SUSPENSION)", "STEERING")
    assert not label_in_text.bracket_match("(ENGINE AND ENGINE COOLING)", "ENGINE")   # a longer name is not the label ENGINE


def test_other_also_matches_the_nhtsa_name_unknown_or_other():
    assert label_in_text.bracket_match("RECALL (UNKNOWN OR OTHER)", "OTHER")
    assert not label_in_text.bracket_match("RECALL (UNKNOWN OR OTHER)", "ENGINE")


def test_the_looser_rule_matches_the_bare_word():
    assert label_in_text.plain_match("THE STEERING WHEEL LOCKED", "STEERING")


def test_summary_counts_and_accuracy_on_a_hand_made_example():
    frame = pd.DataFrame({
        "text": ["RECALL (STEERING) done", "the steering failed", "RECALL (SUSPENSION)", "brakes failed"],
        "label": ["STEERING", "STEERING", "SUSPENSION", "SERVICE BRAKES"],
        "pred": ["STEERING", "ENGINE", "SUSPENSION", "STEERING"],
    })
    result = label_in_text.summarise(frame, label_in_text.bracket_match)
    assert result["n"] == 4 and result["with_label"] == 2 and result["share"] == 0.5
    assert result["accuracy_with"] == 1.0          # both bracket matches are predicted correctly
    assert result["accuracy_without"] == 0.0       # the other two are wrong
    assert label_in_text.summarise(frame, label_in_text.plain_match)["with_label"] == 3
