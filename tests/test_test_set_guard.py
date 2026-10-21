import pytest

from src.prepare import dataset


def test_tuning_data_has_no_test_rows(prepared_dir):
    frames = dataset.load_split("random", prepared_dir)
    assert set(frames) == {"train", "val"}
    assert len(frames["train"]) > len(frames["val"]) > 0


def test_a_system_can_open_the_test_set_only_once(prepared_dir, tmp_path):
    log = tmp_path / "test_access_log.jsonl"
    test = dataset.load_test_once("time", "tfidf_lr", prepared_dir, log)
    assert len(test) > 0
    with pytest.raises(dataset.TestSetAlreadyUsed):
        dataset.load_test_once("time", "tfidf_lr", prepared_dir, log)


def test_other_systems_and_other_splits_have_their_own_access(prepared_dir, tmp_path):
    log = tmp_path / "test_access_log.jsonl"
    dataset.load_test_once("time", "tfidf_lr", prepared_dir, log)
    dataset.load_test_once("random", "tfidf_lr", prepared_dir, log)
    dataset.load_test_once("time", "embedding_lr", prepared_dir, log)
    assert len(dataset.used_tests(log)) == 3


def test_a_forced_repeat_is_allowed_but_logged(prepared_dir, tmp_path):
    log = tmp_path / "test_access_log.jsonl"
    dataset.load_test_once("time", "tfidf_lr", prepared_dir, log)
    dataset.load_test_once("time", "tfidf_lr", prepared_dir, log, force=True)
    assert log.read_text(encoding="utf-8").count('"forced": true') == 1


def test_class_list_ends_with_other(prepared_dir):
    classes = dataset.class_list(prepared_dir)
    assert classes[-1] == "OTHER" and len(classes) == 15
