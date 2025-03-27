from datetime import datetime

from nothingburger.types import Component

class Memory(Component):
    prefix = ""
    suffix = ""
    content = ""

    def __init__(self, **kwargs):
        self.prefix     = kwargs.get('prefix',  self.prefix)
        self.suffix     = kwargs.get('suffix',  self.suffix)

class ConversationalMemory(Memory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []

    def add_message(self, role, content, **kwargs):
        self.messages.append({
            'role':         role,
            'content':      content,
            'timestamp':    kwargs.get('timestamp', datetime.now())
        })

    def add_messages(self, messages, **kwargs):
        for message in messages:
            self.add_message(message['role'], message['content'], **kwargs)