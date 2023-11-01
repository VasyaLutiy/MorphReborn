import os
from datetime import datetime
from llm_dialog import LLMDialog


class ContextFolderDialog(LLMDialog):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def process(self, _):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                time = datetime.fromtimestamp(os.path.getmtime(file_path))
                with open(file_path) as f:
                    file_contents = f.read()
                    self.assign("user", f"Contents for another file \"{file_path}\" in this project:\n\n---\n{file_contents}\n---\n", int(time.timestamp()) * 1000)

        return self.conversation
