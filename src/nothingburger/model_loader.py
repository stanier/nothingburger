#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tomllib

DEFAULTS = {
    "TEMPERATURE": 0.0,
    "TOP_P": 0.1,
    "TOP_K": 40,
    "STOP": ["</stop>"],
    "SEED": -1,
    "MAX_TOKENS": 128,
    "PRESENCE_PENALTY": 1.0,
    "FREQUENCY_PENALTY": 1.0,
    "LOGIT_BIAS": None,
    "LOGPROBS": None,
    "STREAM": False,
    "TYPICAL_P": 0.91,
    "MIROSTAT_MODE": 0,
    "MIROSTAT_TAU": 5.0,
    "MIROSTAT_ETA": 1.0,
    "THREADS": 8,
}

class Adapter:
    def __init__(self, options):
        self.options = options
    
    def _get_generation_param(self, param_name, kwargs, default_key=None):
        """Get generation parameter from kwargs, options, or defaults."""
        default_key = default_key or param_name.upper()
        return kwargs.get(param_name, 
                         self.options.get('generation', {}).get(param_name, 
                                                               DEFAULTS.get(default_key)))

def initializeModel(model_file_path=""):
    options = loadModelFile(model_file_path)
    return loadModel(options)

def loadModelFile(model_file_path):
    with open(model_file_path, "rb") as f:
        return tomllib.load(f)

def loadModel(options):
    provider = options['service']['provider']
    
    if provider == "llama-cpp-python":
        from llama_cpp import Llama
        
        class LlamaCppPythonAdapter(Adapter):
            def __init__(self, options):
                super().__init__(options)
                service = self.options['service']
                self.model = Llama(
                    model_path=service['model_path'],
                    n_gpu_layers=service.get('gpu_layers', 0),
                    n_ctx=service.get('context_length', 2048),
                    main_gpu=service.get('main_gpu', 0),
                    tensor_split=service.get('tensor_split'),
                    **{k: v for k, v in service.get('lora', {}).items() if v is not None},
                    **{k: v for k, v in service.get('rope', {}).items() if v is not None},
                    **{k: v for k, v in service.get('yarn', {}).items() if v is not None},
                )
            
            def generate(self, prompt, **kwargs):
                return self.model(
                    prompt,
                    max_tokens=self._get_generation_param('max_tokens', kwargs),
                    seed=self._get_generation_param('seed', kwargs),
                    top_p=self._get_generation_param('top_p', kwargs),
                    top_k=self._get_generation_param('top_k', kwargs),
                    temperature=self._get_generation_param('temperature', kwargs),
                    **kwargs,
                )
        
        return LlamaCppPythonAdapter(options)
    
    elif provider == "hf_transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        class HfTransformersAdapter(Adapter):
            def __init__(self, options):
                super().__init__(options)
                model_key = self.options['service']['model_key']
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_key, device_map="auto", load_in_4bit=True,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_key)
            
            def generate(self, prompt, **kwargs):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    temperature=self._get_generation_param('temperature', kwargs),
                    top_p=self._get_generation_param('top_p', kwargs),
                    top_k=self._get_generation_param('top_k', kwargs),
                    max_new_tokens=self._get_generation_param('max_tokens', kwargs),
                    **kwargs,
                )
                return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        return HfTransformersAdapter(options)
    
    elif provider == "ctransformers":
        from ctransformers import AutoModelForCausalLM
        
        class CtransformersAdapter(Adapter):
            def __init__(self, options):
                super().__init__(options)
                service = self.options['service']
                self.model = AutoModelForCausalLM.from_pretrained(
                    service['model_path'], model_type=service['model_type']
                )
            
            def generate(self, prompt, **kwargs):
                return self.model.generate(
                    prompt,
                    top_k=self._get_generation_param('top_k', kwargs),
                    top_p=self._get_generation_param('top_p', kwargs),
                    temperature=self._get_generation_param('temperature', kwargs),
                    **kwargs,
                )
        
        return CtransformersAdapter(options)
    
    elif provider == "ollama":
        from ollama import Client as OllamaClient
        from ollama._types import Options as OllamaOptions
        
        class OllamaAdapter(Adapter):
            def __init__(self, options):
                super().__init__(options)
                self.client = OllamaClient(host=self.options['service']['base_url'])
            
            def generate(self, prompt, **kwargs):
                response = self.client.generate(
                    model=kwargs.get('model', self.options['service']['model_key']),
                    prompt=prompt,
                    raw=True,
                    stream=kwargs.get('stream', self._get_generation_param('stream', kwargs, 'STREAM')),
                    options=OllamaOptions(
                        frequency_penalty=self._get_generation_param('frequency_penalty', kwargs),
                        num_predict=self._get_generation_param('max_tokens', kwargs),
                        presence_penalty=self._get_generation_param('presence_penalty', kwargs),
                        seed=self._get_generation_param('seed', kwargs),
                        temperature=self._get_generation_param('temperature', kwargs),
                        top_p=self._get_generation_param('top_p', kwargs),
                        top_k=self._get_generation_param('top_k', kwargs),
                    ),
                )
                return response['response'].strip() if not kwargs.get('stream', False) else response
        
        return OllamaAdapter(options)
    
    elif provider == "openai":
        from openai import OpenAI
        
        class OpenaiAdapter(Adapter):
            def __init__(self, options):
                super().__init__(options)
                service = self.options['service']
                self.client = OpenAI(base_url=service['base_url'], api_key=service['api_key'])
                self.api_format = service.get('api_format', 'chat')
            
            def _convert_to_messages(self, prompt, instruction=None, memory=None):
                """Convert prompt to chat messages format."""
                messages = []
                if instruction:
                    messages.append({"role": "system", "content": instruction})
                if memory and hasattr(memory, 'messages'):
                    for msg in memory.messages:
                        role = "assistant" if msg['role'] in ['assistant', 'Assistant'] else "user"
                        messages.append({"role": role, "content": msg['content']})
                messages.append({"role": "user", "content": prompt})
                return messages
            
            def generate(self, prompt, **kwargs):
                model_key = kwargs.get('model', self.options['service']['model_key'])
                
                if self.api_format == 'chat':
                    messages = self._convert_to_messages(
                        prompt, kwargs.get('instruction'), kwargs.get('memory')
                    )
                    response = self.client.chat.completions.create(
                        model=model_key,
                        messages=messages,
                        max_tokens=self._get_generation_param('max_tokens', kwargs),
                        temperature=self._get_generation_param('temperature', kwargs),
                        top_p=self._get_generation_param('top_p', kwargs),
                        **{k: v for k, v in kwargs.items() 
                           if k in ['presence_penalty', 'frequency_penalty', 'seed', 'stop']}
                    )
                    return response.choices[0].message.content.strip()
                else:
                    response = self.client.completions.create(
                        model=model_key,
                        prompt=prompt,
                        max_tokens=self._get_generation_param('max_tokens', kwargs),
                        temperature=self._get_generation_param('temperature', kwargs),
                        **kwargs,
                    )
                    return response.choices[0].text.strip()
        
        return OpenaiAdapter(options)