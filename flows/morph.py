import os
import re
import openai

from pysyun.conversation.flow.console_bot import ConsoleBot


class MorphBot(ConsoleBot):

    def __init__(self, token):

        super().__init__(token)

        openai.api_key = os.getenv('OPENAI_API_KEY')

    @staticmethod
    def augment_chat(messages):

        total_word_count = 0
        bottom_items = []

        for item in reversed(messages):
            word_count = len(item['content'].split())
            total_word_count += word_count
            if total_word_count < 4097:
                bottom_items.append(item)
            else:
                print("Skipping context", item)

        bottom_items.reverse()

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=bottom_items
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        return result

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

--------------------------------------------------'''

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
            file_name = action['text']
            action["context"].add("generate_file_name", file_name)
            text = f"mrph> Ok, I will create a file \"{file_name}\" when finished. What should be in this file?"
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    @staticmethod
    def build_generate_prompt_input_transition(nested_transition):

        async def transition(action):

            prompt = action['text']

            messages = [{
                "role": "assistant",
                "content": ""
            }, {
                "role": "user",
                "content": prompt
            }]

            response = MorphBot.augment_chat(messages)

            file_name = action["context"].get("generate_file_name")

            # Save the response to a text file
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(response)

            text = f"mrph> Your \"{file_name}\" file was saved."
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)
            await nested_transition(action)

        return transition

    def build_state_machine(self, builder):
        main_menu_transition = self.build_menu_response_transition(
            '''mrph> Welcome to the GPT Morph CLI Bot! You are currently in the main menu.
            
Please choose one of the following options:
1. /generate - Generate a new file for your project.
2. /settings - Display your LLM settings.
3. /graph - Graphviz representation for this bot's API.

To execute a command, type the corresponding option and press Enter.

You can always return to the main menu by typing "/start".''',
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
