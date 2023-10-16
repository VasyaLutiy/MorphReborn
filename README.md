# GPT Morph CLI Bot

The GPT Morph CLI Bot is a command-line interface (CLI) bot powered by OpenAI's GPT-3.5. It allows you to interact with the GPT model to generate text and perform various tasks through a console-based interface.

## Features

- **Text Generation:** You can use this bot to generate text based on your input and project requirements.
- **Settings Display:** Check and display your OpenAI API key settings.
- **Graph Visualization:** View a Graphviz representation of the bot's API.

## Getting Started

To get started with the GPT Morph CLI Bot, follow these steps:

1. Clone this repository to your local machine.

2. Set up your OpenAI API Key:
   - Create a file named ".env" in the project folder.
   - Open the .env file and add the following line:
     ```
     OPENAI_API_KEY=<YOUR_API_KEY>
     ```
     Replace `<YOUR_API_KEY>` with your actual OpenAI API key.
     If you don't have an API key yet, sign up at [OpenAI Platform](https://platform.openai.com/signup).
   - Save the .env file.

3. Run the program:
    ```shell
    python main.py
    ```

4. You will be prompted with a main menu that allows you to choose different options, such as text generation, settings display, and more.

## Usage

- To generate text, select the "/generate" option and follow the prompts.
- To view your OpenAI API key settings, select the "/settings" option.
- Explore other available commands in the main menu.

## BASH Bot API graph
![GPT Morph CLI Bot API graph](./flows/morph.png)
[morph.dot](./flows/morph.dot)

## Contributions

Contributions to this project are welcome. Feel free to open issues, submit pull requests, or suggest improvements.

## License

This project is licensed under the LGPL License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses OpenAI's GPT-3.5 for text generation.

Enjoy using the GPT Morph CLI Bot!
