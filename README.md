# Nothingburger

Nothing but semantic burgers made by humans to feed machines

The goal for `nothingburger` is to create a minimalistic platform in which humans can cook up quick little ~~snacks~~ stacks of language and data that humans can understand and machines can digest.  In other words, it is meant as a convenient solution to the otherwise slightly inconvenient/time-consuming process of cooking up simple recipes for interfacing with Large Language Models and the like.

## Installation

```sh
pip install nothingburger

cp -R $NOTHINGBURGER_SRC/.model_library ./ # this will change later
```

## Usage

### Simple text generation

```python
from nothingburger.model_loader import initializeModel

model = initializeModel(args.model_library + '/' + args.model_file)
print(model.generate("How much wood could a woodchuck chuck if a woodchuck could chuck wood?"))
```

### Example Chain (Interactive Chat)

```python
import nothingburger.templates as templates
import nothingburger.instructions as instructions

from nothingburger.cli import repl
from nothingburger.chains import Chain
from nothingburger.parsers import OutputParser
from nothingburger.model_loader import initializeModel

prompt      = instructions.getRenderedInstruction("chat")
template    = templates.getTemplate("alpaca_instruct_chat")

model_library   = "./.model_library"
model_file      = "ollama/vicuna.toml"

model = initializeModel(model_library + '/' + model_file)

chain = Chain(
    instruction         = prompt,
    template            = template,
    output_parser       = OutputParser(),
    model               = model,
    debug               = False,
    assistant_prefix    = "Assistant: ",
    user_prefix         = "User: ",
)

repl(chain)
``` 

The syntax will likely feel like a much more minimalistic version of LangChain.  Part of the mission is to incorporate an alternate interpretation of LangChain's syntax without all the fluff.

## Modelfiles

One key feature of `nothingburger` is that it enables you to load or connect to LLMs easily through various APIs or libraries by simply writing a TOML file.  For example, here's the entire contents of the file used to load Mistral 7B through Ollama:

```toml
name = "Mistral 7B"
author = "MistralAI"
license = "Apache 2.0"
website = "mistral.ai"

[service]
provider = "ollama"
base_url = "http://localhost:11434"
model_key = "mistral:7b"

[generation]
temperature = 0.7
top_k = 20
top_p = 0.9
max_tokens = 512
seed = 42
batch = 1
repeat_penalty = 1.15
presence_penalty = 0.0
frequency_penalty = 0.0
threads = 0

[generation.mirostat]
mode = 0
eta = 0.1
tau = 5.0
```

The generation values serve as defaults that can you can override at runtime.

## Supported Model Providers/Backends

* Local
  * HuggingFace Transformers
  * Llama.cpp (llama-cpp-python and ctransformers)
  * Ollama
  * HuggingFace Text-Generation-Inference
* Hosted
  * OpenAI (+ Huggingface Endpoints and any other OpenAI-compatible API)
