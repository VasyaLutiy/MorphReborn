import os
import re

from pysyun.conversation.flow.console_bot import ConsoleBot


class RefactorBot(ConsoleBot):

    def __init__(self, token):
        super().__init__(token)

    @staticmethod
    def build_settings_transition():

        openai_api_key = os.getenv("OPENAI_API_KEY")

        async def transition(action):

            if openai_api_key:
                text = f"Your OpenAI API key (.env): \"{openai_api_key}\""
            else:
                text = '''mrph>
                --------------------------------------------------
                    OPENAI API KEY NOT FOUND

                Please set your OpenAI API key by following these steps:

                1. Create a file named ".env" in the project folder.
                2. Open the .env file and add the following line:
                      OPENAI_API_KEY=<YOUR_API_KEY>
                   Replace <YOUR_API_KEY> with your actual OpenAI API key.

                   If you don't have an API key yet, sign up at https://platform.openai.com/signup

                3. Save the .env file.

                Once the OpenAI API key is added, you can proceed with running the program.

                --------------------------------------------------
                '''

            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    @staticmethod
    def build_generate_transition():

        async def transition(action):

            text = "mrph> Enter the file name for saving the generated file:"
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    @staticmethod
    def build_generate_file_name_input_transition():

        async def transition(action):

            text = f"mrph> Ok, I will create a file \"{action['text']}\" when finished. What should be in this file?"
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    @staticmethod
    def build_generate_prompt_input_transition(nested_transition):

        async def transition(action):

            text = f"mrph> Your \"\" file was saved."
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

            await nested_transition(action)

        return transition

    def build_state_machine(self, builder):
        main_menu_transition = self.build_menu_response_transition(
            "mrph> Welcome to the GPT Morph CLI!",
            [["Analyze", "Generate", "Patch"], ["Settings", "Help"]])

        return builder \
            .edge(
                "/start",
                "/start",
                "/graph",
                on_transition=self.build_graphviz_response_transition()) \
            .edge("/start", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/start", "/start", "/settings", on_transition=self.build_settings_transition()) \
            .edge("/start", "/start", "/help") \
            .edge("/start", "/analyze", "/analyze") \
            .edge("/analyze", "/start", "/start") \
            .edge("/start", "/generate_file_name_input", "/generate", on_transition=self.build_generate_transition()) \
            .edge("/generate_file_name_input", "/start", "/start") \
            .edge(
                "/generate_file_name_input",
                "/generate_prompt_input",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_generate_file_name_input_transition()) \
            .edge(
                "/generate_prompt_input",
                "/start",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_generate_prompt_input_transition(main_menu_transition)) \
            .edge("/generate_prompt_input", "/start", "/start")
