from pysyun_chain import Chainable
from ollama_processor import OllamaProcessor


class Source:

    def __init__(self, messages):
        self.messages = messages

    def process(self, _):
        return self.messages


class Console:

    def process(self, data):
        print(data)


message_source = Chainable(Source([{
  "role": "user",
  "content": "Hi! I'm your radio-electronics developer!"
}, {
  "role": "user",
  "content": "Please, generate and Arduino sketch to send a JSON over Bluetooth."
}]))

# llama2
# wizard-vicuna-uncensored:30b
ollama_processor = Chainable(OllamaProcessor('https://api_ollama.detdom.work/api/generate', 'llama2'))

print((message_source | ollama_processor | Chainable(Console())).process([]))
