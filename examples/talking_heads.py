import os
import argparse

from dotenv import load_dotenv

import nothingburger.templates as templates
import nothingburger.instructions as instructions

from nothingburger.cli import repl
from nothingburger.chains import ChatChain
from nothingburger.memory import ConversationalMemory
from nothingburger.parsers import OutputParser
from nothingburger.model_loader import initializeModel

debug = False

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--model-library', default = os.environ.get('BURGER_MODEL_LIBRARY', './.model_library'))
parser.add_argument('--model-file', default = os.environ.get('BURGER_MODEL_FILE', 'ollama/nous-hermes.toml'))
parser.add_argument('--debug', action = 'store_true', help = 'Enable debugging mode')
parser.add_argument('--verbose', action = 'store_true', help = 'Provide extra output')

args = parser.parse_args()

prompt      = instructions.getRenderedInstruction("chat")
template    = templates.getTemplate("alpaca_instruct_chat")

model_library   = "./.model_library"
model_file      = "ollama/vicuna.toml"

model = initializeModel(model_library + '/' + model_file)

class TalkingHeadsChain(ChatChain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actors = [{
            'prefix': "Orc",
            'prompt': "You are an orc undercover at a tavern, pretending to be a dwarf.  Input describes the conversation you have been having with fellow tavern patrons thus far.  Write a response to appropriately continue this conversation while maintaining your cover.",
            'memory': ConversationalMemory(),
        }, {
            'prefix': "Dwarf",
            'prompt': "You are a dwarf undercover at a tavern, pretending to be an orc.  Input describes the conversation you have been having with fellow tavern patrons thus far.  Write a response to appropriately continue this conversation while maintaining your cover.",
            'memory': ConversationalMemory(),
        }]

        self.stop = [self.actors[0]['prefix'] + ": ", self.actors[1]['prefix'] + ": ", "\n"]
        #stop = ["\n"]

        self.last_message = 'How do you do fellow tavern-goer?'
        self.actors[1]['memory'].add_message(self.actors[1]['prefix'], self.last_message)

    def run(self, **kwargs):
        self.last_message = self.generate(
            self.last_message,
            memory = self.actors[0]['memory'],
            instruction = self.actors[0]['prompt'],
            assistant_prefix = self.actors[0]['prefix'],
            user_prefix = self.actors[1]['prefix'],
            max_tokens = 512,
        )
        print(self.actors[0]['prefix'] + ': ' + self.last_message)

        self.last_message = self.generate(
            self.last_message,
            memory = self.actors[1]['memory'],
            instruction = self.actors[0]['prompt'],
            assistant_prefix = self.actors[0]['prefix'],
            user_prefix = self.actors[1]['prefix'],
            max_tokens = 512,
        )
        print(self.actors[1]['prefix'] + ': ' + self.last_message)

chain = TalkingHeadsChain(
    template            = template,
    output_parser       = OutputParser(),
    model               = model,
)

print(chain.actors[1]['prefix'] + ': ' + chain.last_message)
while True:
    chain.run()