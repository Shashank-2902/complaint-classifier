"""A chat client for any provider with an OpenAI-style API (Z.ai, DeepSeek, and others)."""
import httpx


class ApiError(Exception):
    """The provider answered with an error. status_code lets the retry logic tell a rate limit (429) from other errors."""

    def __init__(self, status_code, text):
        super().__init__(f"HTTP {status_code}: {text[:300]}")
        self.status_code = status_code


class OpenAICompatChat:
    def __init__(self, base_url, api_key, model, extra_body=None, timeout=120, client=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.extra_body = extra_body
        self.client = client or httpx.Client(timeout=timeout)

    def _post(self, body):
        try:
            response = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.TimeoutException:
            raise ApiError(408, "the request timed out")
        except httpx.TransportError as error:
            raise ApiError(503, f"network problem: {error}")
        if response.status_code != 200:
            raise ApiError(response.status_code, response.text)
        return response.json()

    def complete(self, messages):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400}
        if self.extra_body:
            body.update(self.extra_body)
        try:
            data = self._post(body)
        except ApiError as error:
            # Some models do not accept the extra setting (for example "thinking"). Try once without it.
            if error.status_code == 400 and self.extra_body:
                self.extra_body = None
                return self.complete(messages)
            raise

        usage = data.get("usage") or {}
        return {
            "content": (data["choices"][0]["message"].get("content") or ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
