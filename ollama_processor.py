import time

from ollama import Client
from typing import List, Dict


class OllamaProcessor:

    def __init__(self, uri: str, model: str):
        self.uri = uri
        self.model = model

    def process(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        client = Client(host=self.uri, verify=False)
        stream = client.chat(
            model=self.model,
            messages=messages,
            stream=True
        )

        results = []
        value = ''
        for chunk in stream:
            value += chunk['message']['content']

        result = {
            'time': int(time.time() * 1000),
            'value': value
        }
        results.append(result)

        return results
