import os
import time

import openai
from openai.lib.azure import AzureOpenAI
from typing import List, Dict


class OpenAIProcessor:
    """
    Client for the OpenAI (or Azure OpenAI) chat completions API.

    The model defaults to the ``OPENAI_MODEL_NAME`` environment variable. The
    underlying client is selected from the environment: a plain OpenAI client is
    used when ``OPENAI_API_KEY`` is set, otherwise an Azure OpenAI client is used
    when ``AZURE_OPENAI_ENDPOINT`` is configured (unless ``disable_azure`` is set).
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None,
                 disable_azure: bool = False):
        self.model = model or os.environ.get("OPENAI_MODEL_NAME")
        self.client = self.build_client(disable_azure, api_key=api_key, base_url=base_url)

    @staticmethod
    def build_client(disable_azure=False, api_key=None, base_url=None):
        # Per-instance credentials take precedence over the process-wide environment,
        # so several named OpenAI processors (different keys/models/endpoints) can coexist.
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

        instance = None
        if openai_api_key is not None:
            client_kwargs = {"api_key": openai_api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            instance = openai.OpenAI(**client_kwargs)
        elif azure_openai_endpoint is not None and not disable_azure:
            instance = AzureOpenAI(
                api_version=azure_openai_api_version,
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
            )

        return instance

    def process(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        value = ''
        for choice in response.choices:
            value += choice.message.content

        result = {
            'time': int(time.time() * 1000),
            'value': value
        }

        return [result]
