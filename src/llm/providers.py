"""The zero-shot providers we can run. Each one becomes its own row in the results table."""
import os

from src.llm.groq_zero_shot import GroqChat
from src.llm.openai_compat import OpenAICompatChat

PROVIDERS = {
    "groq": {
        "system": "groq_zero_shot",
        "key_variable": "GROQ_API_KEY",
        "model_variable": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-20b",
    },
    "zai": {
        "system": "zai_glm_zero_shot",
        "key_variable": "ZAI_API_KEY",
        "model_variable": "ZAI_MODEL",
        "default_model": "glm-4.7-flash",
        "base_url": "https://api.z.ai/api/paas/v4/",
        # GLM models think before they answer by default. For a one-word answer that only costs time.
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}


def model_for(provider):
    settings = PROVIDERS[provider]
    return os.getenv(settings["model_variable"], settings["default_model"])


def make_chat(provider, model):
    settings = PROVIDERS[provider]
    api_key = os.environ[settings["key_variable"]]
    if provider == "groq":
        return GroqChat(api_key, model)
    return OpenAICompatChat(settings["base_url"], api_key, model, settings.get("extra_body"))
