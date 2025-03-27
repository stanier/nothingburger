
import os
import argparse

from dotenv import load_dotenv

from nothingburger.model_loader import initializeModel

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--model-library', default = os.environ.get('ELMI_MODEL_LIBRARY', './.model_library'))
parser.add_argument('--model-file', default = os.environ.get('ELMI_MODEL_FILE', 'ollama/nous-hermes.toml'))
parser.add_argument('--debug', action = 'store_true', help = 'Enable debugging mode')
parser.add_argument('--verbose', action = 'store_true', help = 'Provide extra output')

args = parser.parse_args()

model = initializeModel(args.model_library + '/' + args.model_file)
print(model.generate("How much wood could a woodchuck chuck if a woodchuck could chuck wood?"))