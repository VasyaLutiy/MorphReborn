import os

import openai
from openai.lib.azure import AzureOpenAI


def openai_client(disable_azure=False):
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    instance = openai.OpenAI()
    if azure_openai_endpoint is not None and not disable_azure:
        instance = AzureOpenAI(
            api_version=azure_openai_api_version,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key,
        )

    return instance
