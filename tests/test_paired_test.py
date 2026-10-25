from fractions import Fraction

from src.evaluate import paired_test


def test_the_counts_on_a_hand_made_example():
    truth = ["A", "A", "A", "B", "B", "C"]
    first = ["A", "A", "B", "B", "C", "C"]     # right on items 1, 2, 4, 6
    second = ["A", "B", "A", "C", "C", "B"]    # right on items 1, 3
    both_right, both_wrong, only_first, only_second = paired_test.mcnemar_counts(truth, first, second)
    assert (both_right, both_wrong, only_first, only_second) == (1, 1, 3, 1)


def test_mcnemar_with_three_against_one_is_ten_sixteenths():
    # n = 4. p = 2 * (C(4,0) + C(4,1)) / 2**4 = 2 * (1 + 4) / 16 = 10/16 = 0.625
    numerator, denominator = paired_test.mcnemar_exact(3, 1)
    assert Fraction(numerator, denominator) == Fraction(10, 16)
    assert paired_test.p_text(numerator, denominator) == "0.6250"


def test_mcnemar_five_against_nothing_is_two_thirty_seconds():
    # n = 5. p = 2 * C(5,0) / 2**5 = 2/32 = 0.0625
    numerator, denominator = paired_test.mcnemar_exact(5, 0)
    assert Fraction(numerator, denominator) == Fraction(1, 16)


def test_equal_disagreements_give_one_and_no_disagreement_gives_one():
    for b, c in [(4, 4), (0, 0)]:
        numerator, denominator = paired_test.mcnemar_exact(b, c)
        assert numerator == denominator
    assert paired_test.p_text(*paired_test.mcnemar_exact(0, 0)) == "1.000"


def test_the_test_is_symmetric_in_the_two_systems():
    assert paired_test.mcnemar_exact(7, 2) == paired_test.mcnemar_exact(2, 7)


def test_a_very_small_p_value_is_written_as_a_power_of_ten_without_underflow():
    numerator, denominator = paired_test.mcnemar_exact(5000, 10)
    text = paired_test.p_text(numerator, denominator)
    assert " x 10^-" in text and int(text.split("^")[1]) < -1000
