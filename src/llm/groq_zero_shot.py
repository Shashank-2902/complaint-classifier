"""Zero-shot classification with a Groq model: the class list is in the prompt, temperature 0."""
import time

from src.llm import prompt
from src.llm.cache import make_key

# Price in US dollars per 1 million tokens (input, output), from the Groq models page on 2026-10-20.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    # Z.ai price page, checked 2026-10-21. GLM-4.7-Flash is free, FlashX is the paid and faster version.
    "glm-4.5-flash": (0.0, 0.0),
    "glm-4.7-flash": (0.0, 0.0),
    "glm-4.7-flashx": (0.07, 0.40),
}


# Rate limits (429), time-outs (408) and temporary server problems are worth waiting for.
RETRY_STATUS = (408, 429, 500, 502, 503, 504)


class DailyLimitReached(Exception):
    """Groq says the daily request or token limit is used up. Run again later, the cache keeps the progress."""


class GroqChat:
    """The only place that talks to the Groq SDK. Tests replace it with a fake."""

    def __init__(self, api_key, model):
        from groq import Groq   # imported here so the tests do not need a key

        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, messages):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_completion_tokens=400)
        return {
            "content": response.choices[0].message.content or "",
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }


def cost_usd(model, prompt_tokens, completion_tokens):
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def call_with_retry(chat, messages, sleep=time.sleep, tries=5):
    """Call the model. Wait and try again on a rate limit (HTTP 429), stop when a daily limit is hit."""
    for attempt in range(tries):
        try:
            return chat.complete(messages)
        except Exception as error:  # noqa: BLE001  (the SDK has several error classes, we look at the status code)
            status = getattr(error, "status_code", None)
            if status is None and ("Connection" in type(error).__name__ or "Timeout" in type(error).__name__):
                status = 408                                   # the network dropped: treat like a timeout
            if status not in RETRY_STATUS:
                raise
            text = str(error).lower()
            if status == 429 and ("per day" in text or "tokens per day" in text or "requests per day" in text):
                raise DailyLimitReached(str(error))
            if attempt == tries - 1:
                raise
            sleep(10 * (attempt + 1))


def classify_until_done(chat, model, items, classes, cache, wait_seconds=None, sleep=time.sleep, **kwargs):
    """Like classify_items, but when the daily limit is hit it can wait and continue.

    wait_seconds=None: stop with DailyLimitReached (run again later, the cache keeps the progress).
    wait_seconds=300: sleep 5 minutes and try again. Groq frees tokens gradually, so this keeps going
    until every item is answered. Needs the computer to stay awake.
    """
    while True:
        try:
            return classify_items(chat, model, items, classes, cache, sleep=sleep, **kwargs)
        except DailyLimitReached as error:
            done = sum(1 for _, text in items if cache.get(make_key(model, prompt.PROMPT_VERSION, text)))
            print(f"Daily limit reached after {done} of {len(items)} items. {error}", flush=True)
            if wait_seconds is None:
                raise
            sleep(wait_seconds)


def classify_items(chat, model, items, classes, cache, max_rpm=25, sleep=time.sleep, clock=time.monotonic):
    """Classify (odi_no, text) pairs. Returns one record per item, in the same order.

    Items that are already in the cache cost nothing. New calls are spread out so that at most
    max_rpm requests per minute are made.
    """
    records = []
    gap = 60.0 / max_rpm
    last_call = None

    for odi_no, text in items:
        key = make_key(model, prompt.PROMPT_VERSION, text)
        record = cache.get(key)
        if record is None:
            if last_call is not None and clock() - last_call < gap:
                sleep(gap - (clock() - last_call))
            began = clock()
            last_call = began
            answer = call_with_retry(chat, prompt.build_messages(classes, text), sleep)
            record = {"key": key, "odi_no": odi_no, "raw": answer["content"],
                      "prompt_tokens": answer["prompt_tokens"], "completion_tokens": answer["completion_tokens"],
                      "latency_s": round(clock() - began, 3)}
            cache.put(record)
        records.append({**record, "pred": prompt.parse_label(record["raw"], classes)})
    return records
