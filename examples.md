# GPT Morph CLI Bot Examples

## Example #1 - generating a new Python "Hello, world!" application
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

## Example #2 - generating a Rust socket application
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

## Example #3 - fixing a bug in a Python unit test

```text
...
/patch
mrph> Enter the file name to be patched:
tests/test_source.py
mrph> Loaded "tests/test_source.py". How to augment that?
Please, fix the following error in this code: RuntimeError: no running event loop
llm> To fix the "RuntimeError: no running event loop" error, we need to ensure that we are running the asyncio event loop.

Here's the updated code:

'''python
import asyncio
import unittest

from filters import contract_deployment
from source import TransactionsSource


class TestSource(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_initialize(self):
        source = TransactionsSource(
            "wss://dex.binance.org/api/ws",
            "https://bsc-dataseed.binance.org",
            contract_deployment)

        self.assertTrue(source is not None)

    def test_process(self):
        source = TransactionsSource(
            "wss://dex.binance.org/api/ws",
            "https://bsc-dataseed.binance.org",
            contract_deployment)

        async def process():
            data = await source.process()
            print(data)

        self.loop.run_until_complete(process())


if __name__ == '__main__':
    unittest.main()
'''

In this updated code, we:

1. Added the `setUp` method to initialize the event loop.
2. Wrapped the `source.process()` call in an `async` function `process`.
3. Invoked the `process` function using `self.loop.run_until_complete()`.
mrph> File "tests/test_source.py" has been augmented based on your prompt.
```

## Example #4 - code reviews by injecting "TODO:" comments
```text
mrph> Welcome to the GPT Morph CLI Bot! You are currently in the main menu.
To execute a command, type the corresponding option and press Enter.
You can always return to the main menu by typing "/start".
Type "/help" for more.

[['Analyze', 'Generate', 'Patch'], ['Settings', 'Help', 'Exit'], ['Graph']]
/patch
mrph> Enter the file name to be patched:
main.py
mrph> Loaded "main.py". How to augment that?
['/todo (criticize and add comments)', '*']
/todo
llm>  Here is the updated main.py file with some suggested improvements:

---
# TODO: Add module docstring explaining what this module does

print("Hello, world!") 

# TODO: Print output could be improved to be more descriptive
# TODO: Add input validation to handle invalid input
# TODO: Wrap code in function to allow reuse

---
mrph> The file "main.py" has been criticized. Please, review "TODO:" comments.

/exit
```
