import os
from datetime import datetime
from llm_dialog import LLMDialog


class ContextFolderDialog(LLMDialog):

    def __init__(self, path, filter_callback=None):
        super().__init__()
        self.path = path
        self.filter_callback = filter_callback

    def process(self, _):

        dialog = self

        for root, dirs, files in os.walk(self.path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                if self.filter_callback and not self.filter_callback(file_path):
                    continue

                time = datetime.fromtimestamp(os.path.getmtime(file_path))
                with open(file_path, encoding='utf-8') as file:
                    print(f"Folder context file: {file_path}")
                    file_contents = file.read()
                    self.assign("user", f"Contents for another file \"{file_path}\" in this project:\n\n---\n{file_contents}\n---\n", int(time.timestamp()) * 1000)

        return dialog
