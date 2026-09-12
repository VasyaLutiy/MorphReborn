import time

from openai import OpenAI
from typing import List, Dict


class LlamaCppProcessor:
    """
    Client for a local llama.cpp server (``llama-server``), used for local GPT Morph
    inference on the Tesla K80 node (see the sibling ``docker-k80-nodes`` project).

    That node runs llama.cpp built from source for Kepler (sm_37) and pools the two
    GK210 dies of a single Tesla K80 board (GPU0 + GPU1, 12 GB + 12 GB) behind one
    OpenAI-compatible endpoint at ``http://192.168.0.14:8080/v1`` (model alias
    ``k80-model``). llama-server ignores the API key, but the OpenAI SDK requires a
    non-empty one, hence the ``sk-no-key-required`` default.
    """

    def __init__(self, uri: str, model: str, api_key: str = "sk-no-key-required"):
        self.uri = uri
        self.model = model
        self.client = OpenAI(base_url=uri, api_key=api_key)

    def process(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        results = []
        value = ''
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                value += delta

        result = {
            'time': int(time.time() * 1000),
            'value': value
        }
        results.append(result)

        return results
