<img src="./mrph.png" style="float: right; width: 250px;" />

# GPT Morph CLI Bot

The GPT Morph CLI Bot is a command-line interface (CLI) bot powered by OpenAI's GPT-3.5 or Anthropic's Claude. It allows you to interact with the GPT model to generate text and perform various tasks through a console-based interface.

## Features

- **Text Generation:** You can use this bot to generate text based on your input and project requirements.
- **Settings Display:** Check and display your OpenAI API key settings.
- **Graph Visualization:** View a Graphviz representation of the bot's API.

## Installing the GPT Morph CLI
```shell
pip install git+https://github.com/pysyun/python_gptmorph_cli.git
```

## Getting Started

To get started with the **GPT Morph CLI Bot**, follow these steps:

1. Install it using PIP as described above.

2. Navigate in BASH (it can be your IDE's console) to your project.

3. Configure the Large Language Model (LLM, ChatGPT, Claude, ...).

    2.1. To use OpenAI for morphing, you need to have the OpenAI API Key in the ".env" file.
    
    2.2. Or, set up your OpenAI API Key for a new project:

        - Create a file named ".env" in the project folder.
        - Open the .env file and add the following line:
          ```
          OPENAI_API_KEY=<YOUR_API_KEY>
          ```
        - Replace `<YOUR_API_KEY>` with your actual OpenAI API key.
        - Save the .env file.

    If you don't have an OpenAI API key yet, sign up at [OpenAI Platform](https://platform.openai.com/signup).

    2.3. Or, authenticate to **Anthropic Claude API** by using the [Python Claude Web Authenticator](https://github.com/pysyun/python_claude_web_authenticator).

        - Start the "mrph" shell.
        - Type "/settings".
        - Type "/authenticate_claude".
        - Proceed to signing into https://claude.ai/ to get your API key automatically on successful sign-up.

4. Run the bot:
    ```shell
    mrph
    ```

5. You will be prompted with a main menu that allows you to choose different options, such as text generation, settings display, and more.

## Usage

- To analyze project code, choose the "**/analyze**" option and follow the prompts.
- To generate new code artifacts, select the "**/generate**" option and follow the prompts.
- To modify existing code artifacts, select the "**/patch**" option and follow the prompts.
- To view your LLM key settings, select the "**/settings**" option.
- Explore other available commands in the main menu.

You can find example bot sessions, showing how to do something good at: 
[GPT Morph CLI Bot Examples](./examples.md)

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
