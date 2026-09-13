import os
import time

from typing import List, Dict


class AnthropicProcessor:
    """
    Client for the Anthropic Messages API, the synchronous sibling of the
    OpenAI / llama.cpp / Ollama processors.

    Mirrors their contract: :meth:`process` takes an ``LLMDialog``-shape message
    list and returns ``[{"time", "value"}]``, so the registry's ``run`` picks up
    the text uniformly. The model defaults to ``ANTHROPIC_MODEL_NAME`` (then
    ``claude-sonnet-5``). Per-instance credentials take precedence over the
    process-wide ``ANTHROPIC_API_KEY`` so several named Anthropic processors can
    coexist. The ``anthropic`` SDK is imported lazily inside :meth:`process`, so
    the package is not required merely to import this module.
    """

    def __init__(self, model: str = None, api_key: str = None, max_tokens: int = 8192):
        self.model = model or os.environ.get("ANTHROPIC_MODEL_NAME") or "claude-sonnet-5"
        self.api_key = api_key
        self.max_tokens = max_tokens

    def process(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        value = ''
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                value += text

        result = {
            'time': int(time.time() * 1000),
            'value': value
        }

        return [result]
