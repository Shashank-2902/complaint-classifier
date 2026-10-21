import pytest

from src.llm import groq_zero_shot, prompt
from src.llm.cache import JsonlCache, make_key

CLASSES = ["AIR BAGS", "ENGINE", "ENGINE AND ENGINE COOLING", "STEERING", "OTHER"]


# ---------- parsing the answer ----------

@pytest.mark.parametrize("raw,expected", [
    ("ENGINE", "ENGINE"),
    ("  air bags \n", "AIR BAGS"),
    ('"STEERING".', "STEERING"),
    ("**ENGINE AND ENGINE COOLING**", "ENGINE AND ENGINE COOLING"),
    ("The complaint is about the STEERING system.", "STEERING"),
    ("It is an ENGINE AND ENGINE COOLING problem.", "ENGINE AND ENGINE COOLING"),   # the longer name wins over ENGINE
    ("OTHER", "OTHER"),
])
def test_valid_answers_are_parsed(raw, expected):
    assert prompt.parse_label(raw, CLASSES) == expected


@pytest.mark.parametrize("raw", ["", None, "I am not sure", "BRAKES", "AIR BAGS or STEERING", "   "])
def test_other_answers_are_invalid(raw):
    assert prompt.parse_label(raw, CLASSES) == prompt.INVALID


def test_the_prompt_lists_every_class_and_cuts_long_texts():
    messages = prompt.build_messages(CLASSES, "x" * 5000)
    assert all(name in messages[0]["content"] for name in CLASSES)
    assert len(messages[1]["content"]) == prompt.MAX_TEXT_CHARS


# ---------- the cache ----------

def test_cache_keeps_records_on_disk(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = JsonlCache(path)
    key = make_key("m", "v1", "some text")
    cache.put({"key": key, "odi_no": "1", "raw": "ENGINE"})
    assert JsonlCache(path).get(key)["raw"] == "ENGINE"
    assert JsonlCache(path).get("missing") is None


def test_the_key_changes_with_model_prompt_and_text():
    keys = {make_key("a", "v1", "t"), make_key("b", "v1", "t"), make_key("a", "v2", "t"), make_key("a", "v1", "u")}
    assert len(keys) == 4


# ---------- classification with a fake model (no key, no network) ----------

class FakeChat:
    def __init__(self, answers=None):
        self.calls = 0
        self.answers = answers or {}

    def complete(self, messages):
        self.calls += 1
        text = messages[1]["content"]
        return {"content": self.answers.get(text, "ENGINE"), "prompt_tokens": 100, "completion_tokens": 3}


def run(chat, items, cache, **kwargs):
    return groq_zero_shot.classify_items(chat, "openai/gpt-oss-20b", items, CLASSES, cache, sleep=lambda s: None, **kwargs)


def test_answers_are_parsed_and_returned_in_order(tmp_path):
    chat = FakeChat({"first": "AIR BAGS", "second": "no idea"})
    records = run(chat, [("1", "first"), ("2", "second")], JsonlCache(tmp_path / "c.jsonl"))
    assert [r["pred"] for r in records] == ["AIR BAGS", prompt.INVALID]
    assert records[1]["raw"] == "no idea"          # the raw answer is kept even when it is invalid


def test_cached_items_cost_no_new_calls(tmp_path):
    cache = JsonlCache(tmp_path / "c.jsonl")
    chat = FakeChat()
    run(chat, [("1", "a"), ("2", "b")], cache)
    assert chat.calls == 2
    again = FakeChat()
    run(again, [("1", "a"), ("2", "b")], JsonlCache(tmp_path / "c.jsonl"))      # a new run reads the file
    assert again.calls == 0


def test_requests_are_spread_out_to_respect_the_rate_limit(tmp_path):
    waits, now = [], [0.0]

    def clock():
        return now[0]

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    groq_zero_shot.classify_items(FakeChat(), "openai/gpt-oss-20b", [("1", "a"), ("2", "b"), ("3", "c")], CLASSES,
                                  JsonlCache(tmp_path / "c.jsonl"), max_rpm=30, sleep=sleep, clock=clock)
    assert waits == [2.0, 2.0]                      # 30 requests per minute means 2 seconds between calls


class RateLimit(Exception):
    status_code = 429


def test_a_rate_limit_is_waited_out_and_retried():
    class Flaky(FakeChat):
        def complete(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise RateLimit("rate limit reached")
            return {"content": "ENGINE", "prompt_tokens": 1, "completion_tokens": 1}

    waits = []
    answer = groq_zero_shot.call_with_retry(Flaky(), [], sleep=waits.append)
    assert answer["content"] == "ENGINE" and len(waits) == 2


def test_a_daily_limit_stops_the_run():
    class Daily(FakeChat):
        def complete(self, messages):
            raise RateLimit("Rate limit reached for model: tokens per day (TPD)")

    with pytest.raises(groq_zero_shot.DailyLimitReached):
        groq_zero_shot.call_with_retry(Daily(), [], sleep=lambda s: None)


def test_with_wait_the_run_continues_after_a_daily_limit(tmp_path):
    class DailyThenFine(FakeChat):
        def complete(self, messages):
            self.calls += 1
            if self.calls in (3, 4):
                raise RateLimit("Rate limit reached: tokens per day (TPD), try again in 2m")
            return {"content": "ENGINE", "prompt_tokens": 1, "completion_tokens": 1}

    waits = []
    cache = JsonlCache(tmp_path / "c.jsonl")
    items = [(str(i), f"text {i}") for i in range(5)]
    records = groq_zero_shot.classify_until_done(DailyThenFine(), "openai/gpt-oss-20b", items, CLASSES, cache,
                                                 wait_seconds=300, sleep=waits.append)
    assert len(records) == 5 and all(r["pred"] == "ENGINE" for r in records)
    assert 300 in waits                                  # it slept 5 minutes at the limit and carried on


def test_without_wait_a_daily_limit_stops_but_keeps_the_progress(tmp_path):
    class DailyAfterTwo(FakeChat):
        def complete(self, messages):
            self.calls += 1
            if self.calls > 2:
                raise RateLimit("tokens per day (TPD) reached")
            return {"content": "ENGINE", "prompt_tokens": 1, "completion_tokens": 1}

    cache = JsonlCache(tmp_path / "c.jsonl")
    with pytest.raises(groq_zero_shot.DailyLimitReached):
        groq_zero_shot.classify_until_done(DailyAfterTwo(), "openai/gpt-oss-20b", [(str(i), f"t{i}") for i in range(5)],
                                           CLASSES, cache, wait_seconds=None, sleep=lambda s: None)
    assert len(JsonlCache(tmp_path / "c.jsonl").records) == 2


def test_other_errors_are_not_hidden():
    class Broken(FakeChat):
        def complete(self, messages):
            raise ValueError("bad request")

    with pytest.raises(ValueError):
        groq_zero_shot.call_with_retry(Broken(), [], sleep=lambda s: None)


def test_cost_comes_from_token_counts_and_the_price_list():
    # 1,000,000 input tokens at $0.075 and 1,000,000 output tokens at $0.30
    assert groq_zero_shot.cost_usd("openai/gpt-oss-20b", 1_000_000, 1_000_000) == pytest.approx(0.375)
