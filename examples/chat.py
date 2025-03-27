import os
import argparse

from dotenv import load_dotenv

import nothingburger.templates as templates
import nothingburger.instructions as instructions

from nothingburger.cli import repl
from nothingburger.chains import ChatChain
from nothingburger.parsers import OutputParser
from nothingburger.model_loader import initializeModel

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--model-library', default = os.environ.get('BURGER_MODEL_LIBRARY', './.model_library'))
parser.add_argument('--model-file', default = os.environ.get('BURGER_MODEL_FILE', 'ollama/nous-hermes.toml'))
parser.add_argument('--debug', action = 'store_true', help = 'Enable debugging mode')
parser.add_argument('--verbose', action = 'store_true', help = 'Provide extra output')

args = parser.parse_args()

prompt      = instructions.getRenderedInstruction("chat")
template    = templates.getTemplate("alpaca_instruct_chat_timestamped")

model_library   = "./.model_library"
model_file      = "ollama/vicuna.toml"

model = initializeModel(model_library + '/' + model_file)

chain = ChatChain(
    instruction         = prompt,
    template            = template,
    output_parser       = OutputParser(),
    model               = model,
    debug               = True,
    assistant_prefix    = "Assistant",
    user_prefix         = "User",
    stream              = False,
)

#print(chain.model.tokenize("Hello world!"))

repl(chain)