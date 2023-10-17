# GPT Morph CLI Bot

The GPT Morph CLI Bot is a command-line interface (CLI) bot powered by OpenAI's GPT-3.5. It allows you to interact with the GPT model to generate text and perform various tasks through a console-based interface.

## Features

- **Text Generation:** You can use this bot to generate text based on your input and project requirements.
- **Settings Display:** Check and display your OpenAI API key settings.
- **Graph Visualization:** View a Graphviz representation of the bot's API.

## Installing the GPT Morph CLI
```shell
pip install git+https://github.com/pysyun/python_gptmorph_cli.git
```

## Getting Started

To get started with the GPT Morph CLI Bot, follow these steps:

1. Install it using PIP as stated above.

2. Navigate to a project having the OpenAI API Key in the ".env" file.

3. Or, set up your OpenAI API Key for a new project:
   - Create a file named ".env" in the project folder.
   - Open the .env file and add the following line:
     ```
     OPENAI_API_KEY=<YOUR_API_KEY>
     ```
   - Replace `<YOUR_API_KEY>` with your actual OpenAI API key.
     If you don't have an API key yet, sign up at [OpenAI Platform](https://platform.openai.com/signup).
   - Save the .env file.

4. Run the bot:
    ```shell
    mrph
    ```

5. You will be prompted with a main menu that allows you to choose different options, such as text generation, settings display, and more.

## Usage

- To generate text, select the "/generate" option and follow the prompts.
- To view your OpenAI API key settings, select the "/settings" option.
- Explore other available commands in the main menu.

### Example #1 - generating a new Python "Hello, world!" application
```text
$ mkdir python_hello_world

$ cd python_hello_world/

$ gedit .env

$ cat .env
OPENAI_API_KEY=sk-*********

$ mrph
mrph> Welcome to the GPT Morph CLI Bot! You are currently in the main menu.
            
Please choose one of the following options:
1. /generate - Generate a new file for your project.
2. /settings - Display your LLM settings.
3. /graph - Graphviz representation for this bot's API.

To execute a command, type the corresponding option and press Enter.

You can always return to the main menu by typing "/start".
[['Analyze', 'Generate', 'Patch'], ['Settings', 'Help']]
/generate
mrph> Enter the file name for saving the generated file:
main.py
mrph> Ok, I will create a file "main.py" when finished. What should be in this file?
Please, generate a Python "Hello, world" application.   
mrph> Your "main.py" file was saved.

$ ls
main.py

$ cat main.py 
print("Hello, world!")

$ python main.py
Hello, world!
```

### Example #2 - generating a Rust socket application
```text
$ mkdir rust_hello_socket

$ cd rust_hello_socket/

$ gedit .env

$ cat .env
OPENAI_API_KEY=sk-********

$ mrph
mrph> Welcome to the GPT Morph CLI Bot! You are currently in the main menu.
            
Please choose one of the following options:
1. /generate - Generate a new file for your project.
2. /settings - Display your LLM settings.
3. /graph - Graphviz representation for this bot's API.

To execute a command, type the corresponding option and press Enter.

You can always return to the main menu by typing "/start".
[['Analyze', 'Generate', 'Patch'], ['Settings', 'Help']]
/generate
mrph> Enter the file name for saving the generated file:
main.rs
mrph> Ok, I will create a file "main.rs" when finished. What should be in this file?
Please, generate a Rust application, showing how to use sockets to access https://google.com.
mrph> Your "main.rs" file was saved.

$ ls
main.rs

$ cat main.rs 
use std::io::{Read, Write};
use std::net::TcpStream;
use std::str;

fn main() {
    // Connect to Google's server on port 443 using a TCP stream
    let mut stream = TcpStream::connect("google.com:443").expect("Failed to connect to Google");

    // Send an HTTP GET request to the server
    let request = "GET / HTTP/1.1\r\nHost: google.com\r\n\r\n";
    stream.write_all(request.as_bytes()).expect("Failed to send request");

    // Read the response from the server
    let mut buffer = [0; 2048];
    let response = stream.read(&mut buffer).expect("Failed to read response");

    // Convert the response bytes to a string
    let response_str = str::from_utf8(&buffer[..response]).unwrap();

    // Print the response
    println!("Response from Google:\n{}", response_str);
}
```

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
