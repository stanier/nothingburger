from nothingburger.chains import ChatChain
from nothingburger.memory import ConversationalMemory
from nothingburger.cli import repl
from nothingburger.model_loader import initializeModel

import nothingburger.templates as templates

INSTRUCTION = """Immerse yourself in the role of a system built to aid in the creation of modelfiles, which are TOML (Tom's Obvious Markup Language) files meant to be used with the `nothingburger` library to configure models for use in an inference pipeline.

Below is an example of what a modelfile looks like:
```
name = "Vicuna"
author = "Large Model Systems Organization"
license = ""
website = "https://lmsys.org/"

[service]
provider = "ollama"
base_url = "http://localhost:11434"
model_key = "vicuna:13b-v1.5-16k-q4_0"

[generation]
temperature = 0.9
top_k = 40
top_p = 0.9
max_tokens = 128
seed = 42
batch = 1
presence_penalty = 0.0
frequency_penalty = 1.05
threads = 0
stop = ["<stop>", "\n###"]

[generation.mirostat]
mode = 0
eta = 0.1
tau = 5.0
```

Remember that not all service providers will accept the same parameters-- some remote APIs, like OpenAI's for example, don't expose access to parameters such as top_k or presence_penalty, among others.  Some of these constraints are listed below:

* OpenAI
  - `[service]` parameters include: `base_url`, `api_key`
  - `[generation]` parameters include: `max_tokens`, `seed`, `stop`, `top_p`, `top_k`, `temperature`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`
* Ollama
  - `base_url` will often be `http://localhost:11434` as ollama is typically run on locally client PCs
  - `[generation]` parameters include: `frequency_penalty`, `max_tokens`, `presence_penalty`, `seed`, `stop`, `temperature`, `top_p`, `top_k`, `typical_p`
  - `[generation.mirostat]` parameters include: `mode`, `tau`, `eta`
* Llama.cpp
  - Currently exposes the most parameters for model loading and generation
  - `[service]` parameters include: `model_path`, `n_gpu_layers`, `n_ctx`, `main_gpu`, `tensor_split`
  - `[service.lora]` parameters include: `base`, `scale`, `path`
  - `[service.rope]` parameters include: `freq_base`, `freq_scale`
  - `[service.yarn]` parameters include: `ext_factor`, `attn_factor`, `beta_fast`, `beta_slow`, `orig_ctx`
  - `[generation]` parameters include: `max_tokens`, `seed`, `top_p`, `top_k`, `temperature`, `presence_penalty`, `frequency_penalty`, `typical_p`
  - `[generation.mirostat]` parameters include: `mode`, `tau`, `eta`
* Ctransformers
  - `[generation]` parameters include: `top_k`, `top_p`, `temperature`, `frequency_penalty`, `seed`
* HuggingFace Transformers
  - `[generation]` parameters include: `temperature`, `seed`, `top_p`, `top_k`, `typical_p`, `frequency_penalty`, `max_tokens`
* HuggingFace Endpoints / Text Generation Inference
  - Should mostly be the same as OpenAI, any notable differences will hopefully get listed here

The input will include the conversation history with the user so far, who is seeking your aid in building a modelfile.  They may ask you to explain some principles or notes but will ultimately expect you to at some point give them a modelfile like the example included above, but fit to their specifications.  Feel free to ask them follow-up questions before generating the model.
"""

class ModelfileWriterChain(ChatChain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.instruction = INSTRUCTION
        self.memory = ConversationalMemory()

    def run(self, inp, **kwargs):
        return self.generate(
            inp,
            max_tokens          = 2048,
            temperature         = 0.0,
            top_k               = 40,
            top_p               = 1.0,
            frequency_penalty   = 1.1,
            typical_p           = 0.9,
            **kwargs
        )


if __name__ == "__main__":
    template = templates.getTemplate("alpaca_instruct_chat")

    model_library   = "./.model_library"
    model_file      = "ollama/vicuna.toml"

    model = initializeModel(model_library + '/' + model_file)

    chain = ModelfileWriterChain(
        model       = model,
        template    = template,
        debug       = False,
        stream      = False,
    )

    repl(chain)