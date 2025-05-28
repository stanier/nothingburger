#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tomllib
import json

DEFAULTS = {
    "TEMPERATURE": 0.7,
    "TOP_P": 0.9,
    "TOP_K": 40,
    "STOP": [],
    "SEED": -1,
    "MAX_TOKENS": 1024,
    "PRESENCE_PENALTY": 0.0,
    "FREQUENCY_PENALTY": 0.0,
}

class BaseAdapter:
    """Base adapter with function calling support."""
    
    def __init__(self, options):
        self.options = options
        self.api_format = options.get('service', {}).get('api_format', 'chat')
    
    def _get_generation_param(self, param_name, kwargs, default_key=None):
        """Get generation parameter from kwargs, options, or defaults."""
        default_key = default_key or param_name.upper()
        return kwargs.get(param_name, 
                         self.options.get('generation', {}).get(param_name, 
                                                               DEFAULTS.get(default_key)))

    def convert_tools_to_functions(self, active_tools):
        """Convert roborambo tools to function calling format."""
        if not active_tools:
            return []
            
        functions = []
        for tool_name, tool_instance in active_tools.items():
            for method_name, method_info in tool_instance.methods.items():
                function_def = {
                    "name": f"{tool_name}_{method_name}",
                    "description": method_info.get('description', 'No description'),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                
                # Add parameters from method arguments
                for arg_name, arg_info in method_info.get('arguments', {}).items():
                    arg_type = arg_info.get('arg_type', 'str')
                    
                    # Convert Python types to JSON Schema types
                    type_mapping = {
                        'str': "string",
                        'int': "integer", 
                        'float': "number",
                        'bool': "boolean",
                        'list': "array",
                        'dict': "object"
                    }
                    json_type = type_mapping.get(arg_type, "string")
                    
                    function_def["parameters"]["properties"][arg_name] = {
                        "type": json_type,
                        "description": arg_info.get('arg_desc', 'No description')
                    }
                    
                    # Mark as required if not optional
                    if not arg_info.get('optional', False):
                        function_def["parameters"]["required"].append(arg_name)
                
                functions.append(function_def)
        
        return functions

    def execute_function_call(self, function_call, active_tools):
        """Execute a function call and return the result."""
        try:
            # Handle both dict and object formats
            if hasattr(function_call, 'name'):
                func_name = function_call.name
                args_str = function_call.arguments
            else:
                func_name = function_call.get('name')
                args_str = function_call.get('arguments', '{}')
            
            if '_' not in func_name:
                return f"Invalid function name format: {func_name}"
            
            parts = func_name.split('_', 1)
            tool_name, method_name = parts[0], parts[1]
            
            if tool_name not in active_tools:
                return f"Tool '{tool_name}' not found"
            
            tool_instance = active_tools[tool_name]
            if method_name not in tool_instance.methods:
                return f"Method '{method_name}' not found in tool '{tool_name}'"
            
            # Parse arguments
            try:
                if isinstance(args_str, str):
                    args = json.loads(args_str) if args_str else {}
                else:
                    args = args_str if args_str else {}
            except json.JSONDecodeError:
                return f"Invalid JSON arguments: {args_str}"
            
            # Execute the method
            method = tool_instance.methods[method_name]['method']
            result = method(**args)
            
            return str(result)
            
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"

def initializeModel(model_file_path=""):
    options = loadModelFile(model_file_path)
    return loadModel(options)

def loadModelFile(model_file_path):
    with open(model_file_path, "rb") as f:
        return tomllib.load(f)

def loadModel(options):
    provider = options['service']['provider']
    
    if provider == "openai":
        return OpenAIAdapter(options)
    elif provider == "anthropic":
        return AnthropicAdapter(options)
    elif provider == "ollama":
        return OllamaAdapter(options)
    elif provider == "llama-cpp-python":
        return LlamaCppAdapter(options)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

class OpenAIAdapter(BaseAdapter):
    """OpenAI adapter with conditional thinking spoilers and clean output."""
    
    def __init__(self, options):
        super().__init__(options)
        from openai import OpenAI
        service = self.options['service']
        self.client = OpenAI(
            base_url=service['base_url'], 
            api_key=service['api_key']
        )
    
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
    
    def _is_meaningful_thinking(self, content):
        """Check if content contains meaningful thinking - be more permissive to capture thinking."""
        if not content:
            return False
        
        stripped = content.strip()
        if not stripped or stripped in ["\n\n", "\n", " "]:
            return False
            
        # Don't treat pure function results as thinking
        if stripped.startswith("✅") and len(stripped.split('\n')) == 1:
            return False
            
        # Don't treat JSON as thinking
        if stripped.startswith('{') and stripped.endswith('}'):
            return False
            
        # Be more permissive - if it has some substance, consider it thinking
        if len(stripped) > 10:  # More than just a few characters
            return True
            
        return False
    
    def _clean_response(self, response_text):
        """Remove any raw JSON artifacts from response."""
        if not response_text:
            return ""
        
        response_str = str(response_text).strip()
        
        # Remove JSON artifacts
        import re
        json_patterns = [
            r'\{\s*"tool_calls"\s*:\s*\[\s*\]\s*,\s*"content"\s*:\s*"[^"]*"\s*\}',
            r'\{\s*"tool_calls"\s*:\s*\[\s*\]\s*\}',
        ]
        
        for pattern in json_patterns:
            response_str = re.sub(pattern, '', response_str, flags=re.MULTILINE | re.DOTALL)
        
        # Clean up whitespace
        response_str = re.sub(r'\n\s*\n\s*\n+', '\n\n', response_str).strip()
        return response_str
    
    def generate(self, prompt, **kwargs):
        model_key = kwargs.get('model', self.options['service']['model_key'])
        active_tools = kwargs.get('active_tools', {})
        max_function_calls = kwargs.get('max_function_calls', 15)
        debug = kwargs.get('debug', False)
        
        messages = self._convert_to_messages(
            prompt, kwargs.get('instruction'), kwargs.get('memory')
        )
        
        # Prepare base API parameters
        base_api_params = {
            'model': model_key,
            'max_tokens': self._get_generation_param('max_tokens', kwargs),
            'temperature': self._get_generation_param('temperature', kwargs),
            'top_p': self._get_generation_param('top_p', kwargs),
        }
        
        # Add function calling if tools are available
        if active_tools:
            functions = self.convert_tools_to_functions(active_tools)
            if functions:
                base_api_params['tools'] = [{"type": "function", "function": func} for func in functions]
                base_api_params['tool_choice'] = 'auto'
        
        # Add optional parameters
        for param in ['presence_penalty', 'frequency_penalty', 'seed', 'stop']:
            if param in kwargs:
                base_api_params[param] = kwargs[param]
        
        # Collect thinking and final response
        thinking_parts = []
        function_call_count = 0
        
        while function_call_count < max_function_calls:
            # Make API call
            api_params = {**base_api_params, 'messages': messages}
            
            try:
                response = self.client.chat.completions.create(**api_params)
                message = response.choices[0].message
            except Exception as e:
                return f"Error calling OpenAI API: {str(e)}"
            
            if debug:
                print(f"[DEBUG] Response content: {repr(message.content)}")
                print(f"[DEBUG] Has tool calls: {bool(message.tool_calls)}")
                if message.content:
                    print(f"[DEBUG] Is meaningful thinking: {self._is_meaningful_thinking(message.content)}")
            
            # Check for thinking content before function calls
            if message.tool_calls and self._is_meaningful_thinking(message.content):
                # This is thinking content - the assistant explaining what it's about to do
                thinking_parts.append(message.content.strip())
                if debug:
                    print(f"[DEBUG] Added thinking: {repr(message.content.strip())}")
            
            # If no tool calls, this is our final response
            if not message.tool_calls:
                final_content = self._clean_response(message.content) if message.content else ""
                
                if debug:
                    print(f"[DEBUG] Final response. Thinking parts: {len(thinking_parts)}")
                    print(f"[DEBUG] Final content: {repr(final_content)}")
                
                # Combine thinking + final response using markdown spoilers
                if thinking_parts and final_content:
                    thinking_text = "\n\n".join(thinking_parts)
                    return f"```spoiler Thinking\n{thinking_text}\n```\n\n{final_content}"
                elif thinking_parts:
                    thinking_text = "\n\n".join(thinking_parts)
                    return f"```spoiler Thinking\n{thinking_text}\n```"
                else:
                    # No thinking, just final response
                    return final_content
            
            # Add assistant message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function", 
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            })
            
            # Execute function calls and add results
            for tool_call in message.tool_calls:
                function_result = self.execute_function_call(tool_call.function, active_tools)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": function_result
                })
            
            function_call_count += 1
        
        # If we hit the limit, force a final response
        final_messages = messages + [
            {"role": "user", "content": "Please provide a summary of what you accomplished."}
        ]
        
        try:
            final_response = self.client.chat.completions.create(
                model=model_key,
                messages=final_messages,
                max_tokens=base_api_params['max_tokens'],
                temperature=base_api_params['temperature'],
            )
            
            final_content = self._clean_response(final_response.choices[0].message.content)
            
            # Combine thinking + final response using markdown spoilers
            if thinking_parts and final_content:
                thinking_text = "\n\n".join(thinking_parts)
                return f"```spoiler Thinking\n{thinking_text}\n```\n\n{final_content}"
            elif thinking_parts:
                thinking_text = "\n\n".join(thinking_parts)
                return f"```spoiler Thinking\n{thinking_text}\n```"
            else:
                return final_content or "Function calls completed successfully."
                
        except Exception as e:
            if thinking_parts:
                thinking_text = "\n\n".join(thinking_parts)
                return f"```spoiler Thinking\n{thinking_text}\n```\n\nFunction calls completed successfully."
            else:
                return "Function calls completed successfully."

class AnthropicAdapter(BaseAdapter):
    """Anthropic adapter with function calling."""
    
    def __init__(self, options):
        super().__init__(options)
        try:
            import anthropic
            service = self.options['service']
            self.client = anthropic.Anthropic(api_key=service['api_key'])
        except ImportError:
            raise ImportError("anthropic package required for Anthropic models")
    
    def generate(self, prompt, **kwargs):
        # Anthropic function calling implementation would go here
        model_key = kwargs.get('model', self.options['service']['model_key'])
        active_tools = kwargs.get('active_tools', {})
        
        # For now, basic text generation
        response = self.client.messages.create(
            model=model_key,
            max_tokens=self._get_generation_param('max_tokens', kwargs),
            temperature=self._get_generation_param('temperature', kwargs),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

class OllamaAdapter(BaseAdapter):
    """Ollama adapter with function calling emulation."""
    
    def __init__(self, options):
        super().__init__(options)
        try:
            from ollama import Client as OllamaClient
            self.client = OllamaClient(host=self.options['service']['base_url'])
        except ImportError:
            raise ImportError("ollama package required for Ollama models")
    
    def generate(self, prompt, **kwargs):
        model_key = kwargs.get('model', self.options['service']['model_key'])
        active_tools = kwargs.get('active_tools', {})
        
        # If tools are available, add them to the prompt
        if active_tools:
            functions = self.convert_tools_to_functions(active_tools)
            tools_prompt = self._create_tools_prompt(functions)
            prompt = f"{tools_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        response = self.client.generate(
            model=model_key,
            prompt=prompt,
            options={
                'temperature': self._get_generation_param('temperature', kwargs),
                'top_p': self._get_generation_param('top_p', kwargs),
                'top_k': self._get_generation_param('top_k', kwargs),
            }
        )
        
        response_text = response['response'].strip()
        
        # Try to parse function calls from the response
        if active_tools:
            response_text = self._handle_function_calls_in_text(response_text, active_tools)
        
        return response_text
    
    def _create_tools_prompt(self, functions):
        """Create a prompt that describes available functions."""
        if not functions:
            return ""
        
        tools_desc = "You have access to the following functions. To use them, respond with a JSON object containing 'function_call' with 'name' and 'arguments':\n\n"
        
        for func in functions:
            tools_desc += f"- {func['name']}: {func['description']}\n"
            if func['parameters']['properties']:
                tools_desc += "  Parameters:\n"
                for param, details in func['parameters']['properties'].items():
                    required = " (required)" if param in func['parameters'].get('required', []) else ""
                    tools_desc += f"    - {param} ({details['type']}): {details['description']}{required}\n"
            tools_desc += "\n"
        
        return tools_desc
    
    def _handle_function_calls_in_text(self, response_text, active_tools):
        """Parse and execute function calls from text response."""
        try:
            import re
            json_match = re.search(r'\{[^}]*"function_call"[^}]*\}', response_text)
            if json_match:
                function_call_json = json.loads(json_match.group())
                if 'function_call' in function_call_json:
                    func_call = function_call_json['function_call']
                    result = self.execute_function_call(func_call, active_tools)
                    response_text = response_text.replace(json_match.group(), f"Function result: {result}")
        except:
            pass
        
        return response_text

class LlamaCppAdapter(BaseAdapter):
    """Llama.cpp adapter with function calling emulation."""
    
    def __init__(self, options):
        super().__init__(options)
        try:
            from llama_cpp import Llama
            service = self.options['service']
            self.model = Llama(
                model_path=service['model_path'],
                n_gpu_layers=service.get('gpu_layers', 0),
                n_ctx=service.get('context_length', 2048),
            )
        except ImportError:
            raise ImportError("llama-cpp-python package required for Llama.cpp models")
    
    def generate(self, prompt, **kwargs):
        active_tools = kwargs.get('active_tools', {})
        
        if active_tools:
            functions = self.convert_tools_to_functions(active_tools)
            tools_prompt = self._create_tools_prompt(functions)
            prompt = f"{tools_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        response = self.model(
            prompt,
            max_tokens=self._get_generation_param('max_tokens', kwargs),
            temperature=self._get_generation_param('temperature', kwargs),
            top_p=self._get_generation_param('top_p', kwargs),
            top_k=self._get_generation_param('top_k', kwargs),
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        if active_tools:
            response_text = self._handle_function_calls_in_text(response_text, active_tools)
        
        return response_text
    
    def _create_tools_prompt(self, functions):
        """Same as OllamaAdapter."""
        if not functions:
            return ""
        
        tools_desc = "You have access to the following functions. To use them, respond with a JSON object containing 'function_call' with 'name' and 'arguments':\n\n"
        
        for func in functions:
            tools_desc += f"- {func['name']}: {func['description']}\n"
            if func['parameters']['properties']:
                tools_desc += "  Parameters:\n"
                for param, details in func['parameters']['properties'].items():
                    required = " (required)" if param in func['parameters'].get('required', []) else ""
                    tools_desc += f"    - {param} ({details['type']}): {details['description']}{required}\n"
            tools_desc += "\n"
        
        return tools_desc
    
    def _handle_function_calls_in_text(self, response_text, active_tools):
        """Same as OllamaAdapter."""
        try:
            import re
            json_match = re.search(r'\{[^}]*"function_call"[^}]*\}', response_text)
            if json_match:
                function_call_json = json.loads(json_match.group())
                if 'function_call' in function_call_json:
                    func_call = function_call_json['function_call']
                    result = self.execute_function_call(func_call, active_tools)
                    response_text = response_text.replace(json_match.group(), f"Function result: {result}")
        except:
            pass
        
        return response_text