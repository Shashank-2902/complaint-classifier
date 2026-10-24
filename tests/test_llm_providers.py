import json

import httpx
import pytest

from src.llm import groq_zero_shot, providers
from src.llm.cache import JsonlCache
from src.llm.openai_compat import ApiError, OpenAICompatChat

CLASSES = ["AIR BAGS", "ENGINE", "OTHER"]


def make_chat(handler, **kwargs):
    """A chat client whose HTTP calls go to `handler` instead of the internet."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatChat("https://example.test/api/v4/", "secret-key", "glm-4.5-flash", client=client, **kwargs)


def ok(content="ENGINE", prompt_tokens=300, completion_tokens=2):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                      "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


def test_request_has_model_temperature_zero_key_and_the_extra_setting():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok()

    chat = make_chat(handler, extra_body={"thinking": {"type": "disabled"}})
    answer = chat.complete([{"role": "user", "content": "hi"}])
    assert seen["url"] == "https://example.test/api/v4/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "glm-4.5-flash" and seen["body"]["temperature"] == 0
    assert seen["body"]["thinking"] == {"type": "disabled"}
    assert answer == {"content": "ENGINE", "prompt_tokens": 300, "completion_tokens": 2}


def test_an_empty_answer_becomes_an_empty_string():
    chat = make_chat(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    assert chat.complete([])["content"] == ""


def test_http_errors_carry_the_status_code_so_the_retry_logic_can_read_it():
    chat = make_chat(lambda request: httpx.Response(429, text="too many requests"))
    with pytest.raises(ApiError) as error:
        chat.complete([])
    assert error.value.status_code == 429


def test_a_rate_limit_from_this_client_is_waited_out_and_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="concurrency limit") if len(calls) < 3 else ok("AIR BAGS")

    waits = []
    answer = groq_zero_shot.call_with_retry(make_chat(handler), [], sleep=waits.append)
    assert answer["content"] == "AIR BAGS" and len(waits) == 2


def test_a_model_that_refuses_the_extra_setting_is_retried_without_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(400, text="unknown field thinking") if "thinking" in body else ok()

    chat = make_chat(handler, extra_body={"thinking": {"type": "disabled"}})
    assert chat.complete([])["content"] == "ENGINE"
    assert "thinking" in bodies[0] and "thinking" not in bodies[1]
    chat.complete([])
    assert "thinking" not in bodies[2]                      # it remembers


def test_other_400_errors_are_not_retried_forever():
    chat = make_chat(lambda request: httpx.Response(400, text="bad"))
    with pytest.raises(ApiError):
        chat.complete([])


def test_the_client_works_with_the_classification_pipeline_and_the_cache(tmp_path):
    chat = make_chat(lambda request: ok("air bags"))
    cache = JsonlCache(tmp_path / "c.jsonl")
    records = groq_zero_shot.classify_items(chat, "glm-4.5-flash", [("1", "the air bag did not open")], CLASSES, cache,
                                            sleep=lambda s: None)
    assert records[0]["pred"] == "AIR BAGS"


def test_the_free_glm_models_cost_nothing_and_flashx_has_a_price():
    assert groq_zero_shot.cost_usd("glm-4.5-flash", 1_000_000, 1_000_000) == 0
    assert groq_zero_shot.cost_usd("glm-4.7-flash", 1_000_000, 1_000_000) == 0
    assert groq_zero_shot.cost_usd("glm-4.7-flashx", 1_000_000, 1_000_000) == pytest.approx(0.47)


def test_providers_use_their_own_system_ids_and_keys(monkeypatch):
    assert providers.PROVIDERS["groq"]["system"] == "groq_zero_shot"
    assert providers.PROVIDERS["zai"]["system"] == "zai_glm_zero_shot"
    monkeypatch.setenv("ZAI_API_KEY", "k")
    chat = providers.make_chat("zai", "glm-4.5-flash")
    assert isinstance(chat, OpenAICompatChat) and chat.extra_body == {"thinking": {"type": "disabled"}}
    monkeypatch.delenv("ZAI_MODEL", raising=False)
    assert providers.model_for("zai") == "glm-4.5-flash"


def test_a_timeout_is_reported_as_a_retryable_error():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ApiError) as error:
        make_chat(handler).complete([])
    assert error.value.status_code == 408 and error.value.status_code in groq_zero_shot.RETRY_STATUS


def test_a_timeout_is_waited_out_and_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("too slow", request=request)
        return ok("ENGINE")

    waits = []
    assert groq_zero_shot.call_with_retry(make_chat(handler), [], sleep=waits.append)["content"] == "ENGINE"
    assert len(waits) == 1


def test_a_dropped_connection_is_waited_out_and_retried():
    class APIConnectionError(Exception):
        pass

    calls = []

    class Chat:
        def complete(self, messages):
            calls.append(1)
            if len(calls) == 1:
                raise APIConnectionError("Connection error.")
            return {"content": "ENGINE", "prompt_tokens": 1, "completion_tokens": 1}

    waits = []
    assert groq_zero_shot.call_with_retry(Chat(), [], sleep=waits.append)["content"] == "ENGINE"
    assert len(waits) == 1
