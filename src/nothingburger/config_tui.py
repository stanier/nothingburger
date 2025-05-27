#!/usr/bin/env python3
"""Simple TUI for creating model configuration files."""

import os
import sys
from pathlib import Path

class ModelConfigTUI:
    def __init__(self):
        self.providers = {
            'openai': {
                'name': 'OpenAI',
                'required': ['base_url', 'model_key'],
                'defaults': {
                    'base_url': 'https://api.openai.com/v1',
                    'model_key': 'gpt-4-turbo-preview',
                    'api_format': 'chat'
                }
            },
            'ollama': {
                'name': 'Ollama', 
                'required': ['base_url', 'model_key'],
                'defaults': {
                    'base_url': 'http://localhost:11434',
                    'model_key': 'llama2:7b',
                    'api_format': 'completions'
                }
            },
            'llama-cpp-python': {
                'name': 'Llama.cpp',
                'required': ['model_path'],
                'defaults': {
                    'model_path': './models/model.gguf',
                    'api_format': 'completions'
                }
            }
        }

    def get_input(self, prompt, default=None):
        if default:
            prompt += f" [{default}]"
        result = input(f"{prompt}: ").strip()
        return result if result else default

    def get_confirm(self, prompt, default=True):
        default_str = "Y/n" if default else "y/N"
        result = input(f"{prompt} ({default_str}): ").strip().lower()
        if not result:
            return default
        return result.startswith('y')

    def get_float(self, prompt, default):
        while True:
            try:
                result = input(f"{prompt} [{default}]: ").strip()
                return float(result) if result else default
            except ValueError:
                print("Please enter a valid number.")

    def get_int(self, prompt, default):
        while True:
            try:
                result = input(f"{prompt} [{default}]: ").strip()
                return int(result) if result else default
            except ValueError:
                print("Please enter a valid integer.")

    def configure_provider(self, provider_key):
        provider = self.providers[provider_key]
        config = {
            'service': {
                'provider': provider_key,
                **provider['defaults']
            }
        }
        
        print(f"\nConfiguring {provider['name']}:")
        
        # Get required fields
        for field in provider['required']:
            default = provider['defaults'].get(field, '')
            config['service'][field] = self.get_input(f"{field.replace('_', ' ').title()}", default)
        
        # API format for providers that support it
        if provider_key in ['openai', 'ollama']:
            config['service']['api_format'] = self.get_input("API format", provider['defaults']['api_format'])
        
        return config

    def configure_generation(self, config):
        if not self.get_confirm("Configure generation parameters?", False):
            return config
            
        generation = {}
        generation['temperature'] = self.get_float("Temperature", 0.7)
        generation['max_tokens'] = self.get_int("Max tokens", 512)
        generation['top_p'] = self.get_float("Top P", 0.9)
        generation['top_k'] = self.get_int("Top K (-1 for disabled)", 40)
        
        config['generation'] = generation
        return config

    def get_metadata(self, config):
        print("\nModel metadata:")
        config['name'] = self.get_input("Model name", "My Model")
        config['author'] = self.get_input("Author", "Unknown")
        return config

    def save_config(self, config):
        # Default filename
        provider = config['service']['provider']
        model_name = config.get('name', 'model').lower().replace(' ', '-')
        default_filename = f"{provider}/{model_name}.toml"
        
        filename = self.get_input("Save as", default_filename)
        if not filename.endswith('.toml'):
            filename += '.toml'
        
        # Create directory
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate TOML content manually (no dependency on tomli-w)
        toml_content = self._to_toml(config)
        
        with open(filepath, 'w') as f:
            f.write(toml_content)
        
        print(f"✓ Configuration saved to {filepath}")

    def _to_toml(self, config):
        """Simple TOML generator for basic configs."""
        lines = []
        
        # Top-level fields
        for key, value in config.items():
            if not isinstance(value, dict):
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                else:
                    lines.append(f'{key} = {value}')
        
        # Sections
        for section, values in config.items():
            if isinstance(values, dict):
                lines.append(f'\n[{section}]')
                for key, value in values.items():
                    if isinstance(value, str):
                        lines.append(f'{key} = "{value}"')
                    else:
                        lines.append(f'{key} = {value}')
        
        return '\n'.join(lines) + '\n'

    def run(self):
        print("Model Configuration Generator")
        print("=" * 40)
        
        # Show providers
        print("\nAvailable providers:")
        for key, provider in self.providers.items():
            print(f"  {key}: {provider['name']}")
        
        # Select provider
        provider_key = self.get_input("Select provider", "openai")
        if provider_key not in self.providers:
            print(f"Unknown provider: {provider_key}")
            return
        
        # Configure
        config = self.configure_provider(provider_key)
        config = self.configure_generation(config)
        config = self.get_metadata(config)
        
        # Preview
        print(f"\nGenerated configuration:")
        print("-" * 30)
        print(self._to_toml(config))
        
        # Save
        if self.get_confirm("Save this configuration?", True):
            self.save_config(config)

def main():
    tui = ModelConfigTUI()
    tui.run()

if __name__ == "__main__":
    main()