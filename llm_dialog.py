import datetime
import json


class LLMDialog:

    def __init__(self, speaker_field="role", text_field="content"):
        self.speaker_field = speaker_field
        self.text_field = text_field
        self.conversation = []

    def assign(self, speaker, text, time=None):
        if time is None:
            time = datetime.datetime.now()
            time = int(time.timestamp()) * 1000
        self.conversation.append({
            self.speaker_field: speaker,
            self.text_field: text,
            "time": time
        })

    def json(self):
        return json.dumps(self.conversation)

    def __add__(self, other):
        combined = LLMDialog(speaker_field=self.speaker_field,
                             text_field=self.text_field)

        combined.conversation = self.conversation + other.conversation
        return combined
