#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv

import tomllib

DEFAULTS = {
    "TEMPERATURE"       : 0.0,
    "TOP_P"             : 0.1,
    "TOP_K"             : 40,
    "STOP"              : ["</stop>"],
    "SEED"              : -1,
    "MAX_TOKENS"        : 128,
    "PRESENCE_PENALTY"  : 1.0,
    "FREQUENCY_PENALTY" : 1.0,
    "LOGIT_BIAS"        : None,
    "LOGPROBS"          : None,
    "STREAM"            : False,
    "TYPICAL_P"         : 0.91,
    "MIROSTAT_MODE"     : 0,
    "MIROSTAT_TAU"      : 5.0,
    "MIROSTAT_ETA"      : 1.0,
    "ROPE_FREQ_BASE"    : None,
    "ROPE_FREQ_SCALE"   : None,
    "THREADS"           : 8,
}

class Adapter:
    pass

def initializeModel(model_file_path = "", debug = False):
    options = loadModelFile(model_file_path)
    model = loadModel(options)

    return model

def loadModelFile(model_file_path, debug = False):
    # TODO:  Ensure file exists

    if debug:
        print("Loading modelfile from " + model_file_path)

    with open(model_file_path, "rb") as f:
        data = tomllib.load(f)
        return data

def loadModel(options):
    match options['service']['provider']:
        case "llama-cpp-python":
            from llama_cpp import Llama

            class LlamaCppPythonAdapter(Adapter):
                def __init__(self, options):
                    self.options = options
                    self.model = Llama(
                        model_path          = self.options['service']['model_path'],
                        n_gpu_layers        = self.options['service'].get('gpu_layers'              , 0     ),
                        n_ctx               = self.options['service'].get('context_length'          , 2048  ),
                        main_gpu            = self.options['service'].get('main_gpu'                , 0     ),
                        tensor_split        = self.options['service'].get('tensor_split'            , None  ),
                        lora_base           = self.options['service'].get('lora', {}).get('base'       , None  ),
                        lora_scale          = self.options['service'].get('lora', {}).get('scale'      , 1.0   ),
                        lora_path           = self.options['service'].get('lora', {}).get('path'       , None  ),
                        rope_freq_base      = self.options['service'].get('rope', {}).get('freq_base'  , 0.0   ),
                        rope_freq_scale     = self.options['service'].get('rope', {}).get('freq_scale' , 0.0   ),
                        yarn_ext_factor     = self.options['service'].get('yarn', {}).get('ext_factor' , -1.0  ),
                        yarn_attn_factor    = self.options['service'].get('yarn', {}).get('attn_factor', 1.0   ),
                        yarn_beta_fast      = self.options['service'].get('yarn', {}).get('beta_fast'  , 32.0  ),
                        yarn_beta_slow      = self.options['service'].get('yarn', {}).get('beta_slow'  , 1.0   ),
                        yarn_orig_ctx       = self.options['service'].get('yarn', {}).get('orig_ctx'   , 0     ),
                    )
                
                def count_tokens(self,prompt):
                    return len(self.model.tokenize())

                def generate(self, prompt, **kwargs):
                    output = self.model(
                        prompt,
                        max_tokens          = kwargs.get('max_tokens'       , self.options['generation'].get('max_tokens'       , DEFAULTS["MAX_TOKENS"]        )),
                        seed                = kwargs.get('seed'             , self.options['generation'].get('seed'             , DEFAULTS["SEED"]              )),
                        top_p               = kwargs.get('top_p'            , self.options['generation'].get('top_p'            , DEFAULTS["TOP_P"]             )),
                        top_k               = kwargs.get('top_k'            , self.options['generation'].get('top_k'            , DEFAULTS["TOP_K"]             )),
                        temperature         = kwargs.get('temperature'      , self.options['generation'].get('temperature'      , DEFAULTS["TEMPERATURE"]       )),
                        presence_penalty    = kwargs.get('presence_penalty' , self.options['generation'].get('presence_penalty' , DEFAULTS["PRESENCE_PENALTY"]  )),
                        repeat_penalty      = kwargs.get('frequency_penalty', self.options['generation'].get('frequency_penalty', DEFAULTS["FREQUENCY_PENALTY"] )),
                        typical_p           = kwargs.get('typical_p'        , self.options['generation'].get('typical_p'        , DEFAULTS["TYPICAL_P"]         )),
                        mirostat_mode       = kwargs.get('mirostat_mode'    , self.options.get('generation.mirostat', {}).get('mode'    , DEFAULTS["MIROSTAT_MODE"]     )),
                        mirostat_tau        = kwargs.get('mirostat_tau'     , self.options.get('generation.mirostat', {}).get('tau'     , DEFAULTS["MIROSTAT_TAU"]      )),
                        mirostat_eta        = kwargs.get('mirostat_eta'     , self.options.get('generation.mirostat', {}).get('eta'     , DEFAULTS["MIROSTAT_ETA"]      )),
                        **kwargs,
                    )
                    
                    return output

            return LlamaCppPythonAdapter(options)

        case "hf_transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            class HfTransformersAdapter(Adapter):
                def __init__(self, options):
                    self.options = options
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.options['service']['model_key'],
                        device_map          = "auto",
                        load_in_4bit        = True,
                        torch_dtype         = torch.float16,
                        low_cpu_mem_usage   = True
                    )

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.options['service']['model_path']
                    )

                def count_tokens(self, prompt):
                    return len(self.tokenizer(prompt, return_tensors="pt"))

                def generate(self, prompt, **kwargs):
                    inputs = self.tokenizer(prompt, return_tensors="pt")

                    generate_ids = adapter.model.generate(
                        inputs.input_ids,
                        temperature         = kwargs.get('temperature'      , self.options['generation'].get('temperature'      , DEFAULTS["TEMPERATURE"]       )),
                        seed                = kwargs.get('seed'             , self.options['generation'].get('seed'             , DEFAULTS["SEED"]              )),
                        top_p               = kwargs.get('top_p'            , self.options['generation'].get('top_p'            , DEFAULTS["TOP_P"]             )),
                        top_k               = kwargs.get('top_k'            , self.options['generation'].get('top_k'            , DEFAULTS["TOP_K"]             )),
                        typical_p           = kwargs.get('typical_p'        , self.options['generation'].get('typical_p'        , DEFAULTS["TYPICAL_P"]         )),
                        repetition_penalty  = kwargs.get('frequency_penalty', self.options['generation'].get('frequency_penalty', DEFAULTS["FREQUENCY_PENALTY"] )),
                        max_new_tokens      = kwargs.get('max_tokens'       , self.options['generation'].get('max_tokens'       , DEFAULTS["MAX_TOKENS"]        )),
                        **kwargs,
                    )

                    output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    return output
            
            return HfTransformersAdapter(options)

        case "ctransformers":
            from ctransformers import AutoModelForCausalLM, AutoTokenizer

            class CtransformersAdapter(Adapter):
                def __init__(self, options):
                    self.options = options
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.options['service']['model_path'],
                        model_type = self.options['service']['model_type'],
                    )

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.options['service']['model_path']
                    )

                def count_tokens(self, prompt):
                    return len(self.tokenizer(prompt, return_tensors="pt"))

                def generate(self, prompt, **kwargs):
                    output = self.model.generate(
                        prompt,
                        top_k               = kwargs.get('top_k'            , self.options['generation'].get('top_k'            , DEFAULTS["TOP_P"]             )),
                        top_p               = kwargs.get('top_p'            , self.options['generation'].get('top_p'            , DEFAULTS["TOP_P"]             )),
                        temperature         = kwargs.get('temperature'      , self.options['generation'].get('temperature'      , DEFAULTS["TEMPERATURE"]       )),
                        repetition_penalty  = kwargs.get('frequency_penalty', self.options['generation'].get('frequency_penalty', DEFAULTS["FREQUENCY_PENALTY"] )),
                        seed                = kwargs.get('seed'             , self.options['generation'].get('seed'             , DEFAULTS["SEED"]              )),
                        #threads             = threads               or self.options['generation'].get('threads'),
                        **kwargs,
                    )

                    return output

            return CtransformersAdapter()

        case "ollama":
            from ollama import Client as OllamaClient
            from ollama._types import Options as OllamaOptions
            
            class OllamaAdapter(Adapter):
                def __init__(self, options):
                    self.options = options
                    self.client = OllamaClient(
                        host = self.options['service']['base_url'],
                    )

                def count_tokens(self, prompt):
                    response = self.client.generate(
                        model       = self.options['service']['model_key'],
                        template    = prompt,
                        raw         = True,
                        options     = OllamaOptions(
                            num_predict = 0,
                        )
                    )

                    return response["prompt_eval_count"]

                def tokenize(self, prompt, **kwargs):
                    return self.client.generate(
                        model       = self.options['service']['model_key'],
                        template    = prompt,
                        prompt      = '',
                        raw         = False,
                        options     = OllamaOptions(
                            num_predict = 0,
                        )
                    )['context']

                def generate(self, prompt, **kwargs):
                    response = self.client.generate(
                        model   = kwargs.get('model', self.options['service']['model_key']),
                        prompt  = prompt,
                        raw     = True,
                        stream  = kwargs.get('stream', self.options['generation'].get('stream', DEFAULTS["STREAM"])),
                        options = OllamaOptions(
                            frequency_penalty   = kwargs.get('frequency_penalty'    , self.options['generation'].get('frequency_penalty', DEFAULTS["FREQUENCY_PENALTY"] )),
                            num_predict         = kwargs.get('max_tokens'           , self.options['generation'].get('max_tokens'       , DEFAULTS["MAX_TOKENS"]        )),
                            presence_penalty    = kwargs.get('presence_penalty'     , self.options['generation'].get('presence_penalty' , DEFAULTS["PRESENCE_PENALTY"]  )),
                            seed                = kwargs.get('seed'                 , self.options['generation'].get('seed'             , DEFAULTS["SEED"]              )),
                            stop                = kwargs.get('stop'                 , self.options['generation'].get('stop'             , DEFAULTS["STOP"]              )),
                            temperature         = kwargs.get('temperature'          , self.options['generation'].get('temperature'      , DEFAULTS["TEMPERATURE"]       )),
                            top_p               = kwargs.get('top_p'                , self.options['generation'].get('top_p'            , DEFAULTS["TOP_P"]             )),
                            top_k               = kwargs.get('top_k'                , self.options['generation'].get('top_k'            , DEFAULTS["TOP_K"]             )),
                            num_thread          = kwargs.get('threads'              , self.options['generation'].get('threads'          , DEFAULTS["THREADS"]           )),
                            typical_p           = kwargs.get('typical_p'            , self.options['generation'].get('typical_p'        , DEFAULTS["TYPICAL_P"]         )),
                            mirostat            = kwargs.get('mirostat'             , self.options['generation'].get('mirostat', {}).get('mode', DEFAULTS["MIROSTAT_MODE"]     )),
                            mirostat_eta        = kwargs.get('mirostat_eta'         , self.options['generation'].get('mirostat', {}).get('eta' , DEFAULTS["MIROSTAT_ETA"]      )),
                            mirostat_tau        = kwargs.get('mirostat_tau'         , self.options['generation'].get('mirostat', {}).get('tau' , DEFAULTS["MIROSTAT_TAU"]      )),
                            #rope_frequency_base     = kwargs.get('rope_frequency_base'  , self.options['generation'].get('rope',     {}).get('freq_base' , DEFAULTS["ROPE_FREQ_BASE"])),
                            #rope_frequency_scale    = kwargs.get('rope_frequency_scale' , self.options['generation'].get('rope',     {}).get('freq_scale', DEFAULTS["ROPE_FREQ_SCALE"])),
                        ),
                    )

                    if kwargs.get('stream', False):
                        return response
                    else:
                        return response['response'].strip()

            return OllamaAdapter(options)

        case "openai":
            from openai import OpenAI

            class OpenaiAdapter(Adapter):
                def __init__(self, options):
                    self.options = options
                    self.client = OpenAI(
                        base_url    = self.options['service']['base_url'],
                        api_key     = self.options['service']['api_key'],
                    )
                    # Default to chat format for modern models
                    self.api_format = self.options['service'].get('api_format', 'chat')

                def count_tokens(self, prompt_or_messages):
                    if self.api_format == 'chat':
                        # For chat format, we need to estimate tokens from messages
                        if isinstance(prompt_or_messages, list):
                            # Already in message format
                            total_content = ' '.join([msg.get('content', '') for msg in prompt_or_messages])
                        else:
                            # Convert prompt to messages for estimation
                            total_content = str(prompt_or_messages)
                        # Rough estimation - in production you'd use tiktoken
                        return len(total_content.split()) * 1.3
                    else:
                        # Legacy completions format
                        response = self.client.completions.create(
                            model=self.options['service']['model_key'],
                            prompt=prompt_or_messages,
                            max_tokens=0,
                        )
                        return response.usage.prompt_tokens

                def _convert_prompt_to_messages(self, prompt, instruction=None):
                    """Convert a formatted prompt string to chat messages format."""
                    messages = []
                    
                    if instruction:
                        messages.append({"role": "system", "content": instruction})
                    
                    # For simple prompts, treat as user message
                    if isinstance(prompt, str):
                        messages.append({"role": "user", "content": prompt})
                    
                    return messages

                def _extract_messages_from_memory(self, memory, instruction=None):
                    """Extract messages from conversational memory."""
                    messages = []
                    
                    if instruction:
                        messages.append({"role": "system", "content": instruction})
                    
                    if memory and hasattr(memory, 'messages'):
                        for msg in memory.messages:
                            role = "assistant" if msg['role'] in ['assistant', 'Assistant'] else "user"
                            messages.append({
                                "role": role,
                                "content": msg['content']
                            })
                    
                    return messages

                def generate(self, prompt, **kwargs):
                    model_key = kwargs.get('model', self.options['service']['model_key'])
                    
                    if self.api_format == 'chat':
                        # Modern chat completions API
                        memory = kwargs.get('memory')
                        instruction = kwargs.get('instruction')
                        
                        if memory and hasattr(memory, 'messages') and len(memory.messages) > 0:
                            # Use conversational memory to build messages
                            messages = self._extract_messages_from_memory(memory, instruction)
                            # Add the current prompt as the latest user message
                            messages.append({"role": "user", "content": prompt})
                        else:
                            # Convert prompt to messages format
                            messages = self._convert_prompt_to_messages(prompt, instruction)
                        
                        response = self.client.chat.completions.create(
                            model=model_key,
                            messages=messages,
                            max_tokens=kwargs.get('max_tokens', self.options['generation'].get('max_tokens', DEFAULTS["MAX_TOKENS"])),
                            temperature=kwargs.get('temperature', self.options['generation'].get('temperature', DEFAULTS["TEMPERATURE"])),
                            top_p=kwargs.get('top_p', self.options['generation'].get('top_p', DEFAULTS["TOP_P"])),
                            presence_penalty=kwargs.get('presence_penalty', self.options['generation'].get('presence_penalty', DEFAULTS["PRESENCE_PENALTY"])),
                            frequency_penalty=kwargs.get('frequency_penalty', self.options['generation'].get('frequency_penalty', DEFAULTS["FREQUENCY_PENALTY"])),
                            seed=kwargs.get('seed', self.options['generation'].get('seed', DEFAULTS["SEED"])),
                            stop=kwargs.get('stop', self.options['generation'].get('stop', DEFAULTS["STOP"])),
                        )
                        return response.choices[0].message.content.strip()
                    
                    else:
                        # Legacy completions API
                        response = self.client.completions.create(
                            model=model_key,
                            prompt=prompt,
                            max_tokens=kwargs.get('max_tokens', self.options['generation'].get('max_tokens', DEFAULTS["MAX_TOKENS"])),
                            seed=kwargs.get('seed', self.options['generation'].get('seed', DEFAULTS["SEED"])),
                            stop=kwargs.get('stop', self.options['generation'].get('stop', DEFAULTS["STOP"])),
                            top_p=kwargs.get('top_p', self.options['generation'].get('top_p', DEFAULTS["TOP_P"])),
                            temperature=kwargs.get('temperature', self.options['generation'].get('temperature', DEFAULTS["TEMPERATURE"])),
                            presence_penalty=kwargs.get('presence_penalty', self.options['generation'].get('presence_penalty', DEFAULTS["PRESENCE_PENALTY"])),
                            frequency_penalty=kwargs.get('frequency_penalty', self.options['generation'].get('frequency_penalty', DEFAULTS["FREQUENCY_PENALTY"])),
                            logit_bias=kwargs.get('logit_bias', self.options['generation'].get('logit_bias', DEFAULTS["LOGIT_BIAS"])),
                            logprobs=kwargs.get('logprobs', self.options['generation'].get('logprobs', DEFAULTS["LOGPROBS"])),
                        )
                        return response.choices[0].text.strip()

            return OpenaiAdapter(options)