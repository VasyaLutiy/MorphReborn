import os

import openai
from dotenv import load_dotenv


def load_settings():

    current_directory = os.getcwd()
    env_path = os.path.join(current_directory, '.env')
    load_dotenv(dotenv_path=env_path)

    openai.api_key = os.getenv('OPENAI_API_KEY')
