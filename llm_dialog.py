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
        self.conversation.append({
            self.speaker_field: speaker,
            self.text_field: text,
            "time": int(time.timestamp()) * 1000
        })

    def json(self):
        return json.dumps(self.conversation)

    def __add__(self, other):
        combined = LLMDialog(speaker_field=self.speaker_field,
                             text_field=self.text_field)

        combined.conversation = self.conversation + other.conversation
        return combined
