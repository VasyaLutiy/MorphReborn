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
                text = '''
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

    def build_state_machine(self, builder):
        main_menu_transition = self.build_menu_response_transition(
            "GPT Morph CLI",
            [["Analyze", "Generate"], ["Settings", "Help"]])

        return builder \
            .edge(
                "/start",
                "/start",
                "/graph",
                on_transition=self.build_graphviz_response_transition()) \
            .edge("/start", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/start", "/start", "/settings", on_transition=self.build_settings_transition()) \
            .edge("/start", "/start", None, matcher=re.compile("/help|Help"))
