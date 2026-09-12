import os

import openai
from dotenv import load_dotenv

from processors.registry import ProcessorRegistry


def load_settings():

    current_directory = os.getcwd()
    env_path = os.path.join(current_directory, '.env')
    load_dotenv(dotenv_path=env_path)

    openai.api_key = os.getenv('OPENAI_API_KEY')


def load_registry():
    """Build the named processor registry from the current environment.

    ``.env`` must already be loaded (see :func:`load_settings`). Returns a
    :class:`ProcessorRegistry` describing every configured processor instance.
    """
    return ProcessorRegistry.from_env()
