import os

import openai
from openai.lib.azure import AzureOpenAI


def openai_client(disable_azure=False):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    instance = None
    if openai_api_key is not None:
        instance = openai.OpenAI()
    elif azure_openai_endpoint is not None and not disable_azure:
        instance = AzureOpenAI(
            api_version=azure_openai_api_version,
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_api_key,
        )

    return instance
