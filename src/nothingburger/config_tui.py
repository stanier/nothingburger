#!/usr/bin/env python3
"""
Interactive TUI for creating model configuration files for nothingburger.
"""

import os
import sys
import tomllib
from pathlib import Path

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class ModelConfigTUI:
    def __init__(self):
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None
        
        self.providers = {
            'openai': {
                'name': 'OpenAI',
                'description': 'OpenAI GPT models (GPT-4, GPT-3.5, etc.)',
                'supports_chat': True,
                'supports_completions': True,
                'default_format': 'chat'
            },
            'ollama': {
                'name': 'Ollama',
                'description': 'Local Ollama server',
                'supports_chat': True,
                'supports_completions': True,
                'default_format': 'completions'
            },
            'llama-cpp-python': {
                'name': 'Llama.cpp (Python)',
                'description': 'Direct llama.cpp integration',
                'supports_chat': False,
                'supports_completions': True,
                'default_format': 'completions'
            },
            'hf_transformers': {
                'name': 'HuggingFace Transformers',
                'description': 'HuggingFace transformers library',
                'supports_chat': False,
                'supports_completions': True,
                'default_format': 'completions'
            },
            'ctransformers': {
                'name': 'CTransformers',
                'description': 'C++ transformers backend',
                'supports_chat': False,
                'supports_completions': True,
                'default_format': 'completions'
            }
        }

    def print_header(self):
        if self.console:
            self.console.print(Panel.fit(
                "[bold blue]Nothingburger Model Configuration Generator[/bold blue]\n"
                "Create model configuration files for LLM providers",
                border_style="blue"
            ))
        else:
            print("=" * 60)
            print("Nothingburger Model Configuration Generator")
            print("Create model configuration files for LLM providers")
            print("=" * 60)

    def show_providers(self):
        if self.console:
            table = Table(title="Available Providers")
            table.add_column("Key", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Description", style="green")
            table.add_column("Chat API", style="yellow")
            
            for key, provider in self.providers.items():
                chat_support = "✓" if provider['supports_chat'] else "✗"
                table.add_row(key, provider['name'], provider['description'], chat_support)
            
            self.console.print(table)
        else:
            print("\nAvailable Providers:")
            for key, provider in self.providers.items():
                chat_support = "Yes" if provider['supports_chat'] else "No"
                print(f"  {key}: {provider['name']} - {provider['description']} (Chat API: {chat_support})")

    def get_input(self, prompt, default=None, choices=None):
        if HAS_RICH and choices:
            return Prompt.ask(prompt, choices=choices, default=default)
        elif HAS_RICH:
            return Prompt.ask(prompt, default=default)
        else:
            if choices:
                prompt += f" ({'/'.join(choices)})"
            if default:
                prompt += f" [{default}]"
            result = input(f"{prompt}: ").strip()
            return result if result else default

    def get_float(self, prompt, default=None):
        if HAS_RICH:
            return FloatPrompt.ask(prompt, default=default)
        else:
            if default is not None:
                prompt += f" [{default}]"
            while True:
                try:
                    result = input(f"{prompt}: ").strip()
                    return float(result) if result else default
                except ValueError:
                    print("Please enter a valid number.")

    def get_int(self, prompt, default=None):
        if HAS_RICH:
            return IntPrompt.ask(prompt, default=default)
        else:
            if default is not None:
                prompt += f" [{default}]"
            while True:
                try:
                    result = input(f"{prompt}: ").strip()
                    return int(result) if result else default
                except ValueError:
                    print("Please enter a valid integer.")

    def get_confirm(self, prompt, default=True):
        if HAS_RICH:
            return Confirm.ask(prompt, default=default)
        else:
            default_str = "Y/n" if default else "y/N"
            result = input(f"{prompt} ({default_str}): ").strip().lower()
            if not result:
                return default
            return result.startswith('y')

    def configure_openai(self):
        config = {
            'service': {
                'provider': 'openai',
                'base_url': self.get_input("Base URL", "https://api.openai.com/v1"),
                'model_key': self.get_input("Model name", "gpt-4-turbo-preview"),
            }
        }
        
        # API format selection
        supports_chat = True
        api_format = self.get_input("API format", "chat", ["chat", "completions"])
        config['service']['api_format'] = api_format
        
        if self.get_confirm("Do you want to set an API key now?", False):
            config['service']['api_key'] = self.get_input("API key")
        
        return config

    def configure_ollama(self):
        config = {
            'service': {
                'provider': 'ollama',
                'base_url': self.get_input("Ollama server URL", "http://localhost:11434"),
                'model_key': self.get_input("Model name", "llama2:7b"),
            }
        }
        
        api_format = self.get_input("API format", "completions", ["chat", "completions"])
        config['service']['api_format'] = api_format
        
        return config

    def configure_llama_cpp(self):
        config = {
            'service': {
                'provider': 'llama-cpp-python',
                'model_path': self.get_input("Path to model file", "./models/model.gguf"),
                'api_format': 'completions'
            }
        }
        
        if self.get_confirm("Configure advanced settings?", False):
            config['service']['n_gpu_layers'] = self.get_int("GPU layers", 0)
            config['service']['n_ctx'] = self.get_int("Context length", 2048)
            config['service']['main_gpu'] = self.get_int("Main GPU", 0)
        
        return config

    def configure_hf_transformers(self):
        config = {
            'service': {
                'provider': 'hf_transformers',
                'model_key': self.get_input("HuggingFace model name", "microsoft/DialoGPT-medium"),
                'api_format': 'completions'
            }
        }
        return config

    def configure_ctransformers(self):
        config = {
            'service': {
                'provider': 'ctransformers',
                'model_path': self.get_input("Path to model file", "./models/model.bin"),
                'model_type': self.get_input("Model type", "llama"),
                'api_format': 'completions'
            }
        }
        return config

    def configure_generation_settings(self, config):
        if self.console:
            self.console.print("\n[bold]Generation Settings[/bold]")
        else:
            print("\nGeneration Settings:")
        
        if self.get_confirm("Configure generation parameters?", True):
            generation = {}
            
            generation['temperature'] = self.get_float("Temperature", 0.7)
            generation['max_tokens'] = self.get_int("Max tokens", 512)
            generation['top_p'] = self.get_float("Top P", 0.9)
            generation['top_k'] = self.get_int("Top K (-1 for disabled)", 40)
            generation['seed'] = self.get_int("Seed (-1 for random)", -1)
            generation['presence_penalty'] = self.get_float("Presence penalty", 0.0)
            generation['frequency_penalty'] = self.get_float("Frequency penalty", 0.0)
            
            # Stop sequences
            if self.get_confirm("Add stop sequences?", False):
                stop_sequences = []
                while True:
                    stop = self.get_input("Stop sequence (empty to finish)")
                    if not stop:
                        break
                    stop_sequences.append(stop)
                if stop_sequences:
                    generation['stop'] = stop_sequences
            
            config['generation'] = generation
            
            # Mirostat settings for supported providers
            if config['service']['provider'] in ['ollama', 'llama-cpp-python']:
                if self.get_confirm("Configure Mirostat sampling?", False):
                    mirostat = {}
                    mirostat['mode'] = self.get_int("Mirostat mode", 0, choices=[0, 1, 2])
                    if mirostat['mode'] > 0:
                        mirostat['tau'] = self.get_float("Mirostat tau", 5.0)
                        mirostat['eta'] = self.get_float("Mirostat eta", 0.1)
                    config['generation']['mirostat'] = mirostat

    def get_model_metadata(self):
        if self.console:
            self.console.print("\n[bold]Model Metadata[/bold]")
        else:
            print("\nModel Metadata:")
        
        metadata = {}
        metadata['name'] = self.get_input("Model display name", "My Model")
        metadata['author'] = self.get_input("Author/Organization", "Unknown")
        metadata['license'] = self.get_input("License", "")
        metadata['website'] = self.get_input("Website/URL", "")
        
        return metadata

    def generate_config(self):
        self.print_header()
        self.show_providers()
        
        # Select provider
        provider_key = self.get_input(
            "\nSelect provider", 
            "openai", 
            list(self.providers.keys())
        )
        
        if provider_key not in self.providers:
            print(f"Invalid provider: {provider_key}")
            return None
        
        provider = self.providers[provider_key]
        
        if self.console:
            self.console.print(f"\n[bold]Configuring {provider['name']}[/bold]")
        else:
            print(f"\nConfiguring {provider['name']}")
        
        # Configure based on provider
        if provider_key == 'openai':
            config = self.configure_openai()
        elif provider_key == 'ollama':
            config = self.configure_ollama()
        elif provider_key == 'llama-cpp-python':
            config = self.configure_llama_cpp()
        elif provider_key == 'hf_transformers':
            config = self.configure_hf_transformers()
        elif provider_key == 'ctransformers':
            config = self.configure_ctransformers()
        else:
            print(f"Provider {provider_key} not yet implemented")
            return None
        
        # Add metadata
        metadata = self.get_model_metadata()
        config.update(metadata)
        
        # Configure generation settings
        self.configure_generation_settings(config)
        
        return config

    def save_config(self, config):
        if self.console:
            self.console.print("\n[bold]Save Configuration[/bold]")
        else:
            print("\nSave Configuration")
        
        # Default filename
        provider = config['service']['provider']
        model_name = config.get('name', 'model').lower().replace(' ', '-')
        default_filename = f"{provider}/{model_name}.toml"
        
        filename = self.get_input("Filename", default_filename)
        
        # Ensure .toml extension
        if not filename.endswith('.toml'):
            filename += '.toml'
        
        # Create directory if needed
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to TOML format
        import tomli_w
        
        try:
            with open(filepath, 'wb') as f:
                tomli_w.dump(config, f)
            
            if self.console:
                self.console.print(f"[green]✓ Configuration saved to {filepath}[/green]")
            else:
                print(f"✓ Configuration saved to {filepath}")
            
            return str(filepath)
        except Exception as e:
            if self.console:
                self.console.print(f"[red]✗ Error saving configuration: {e}[/red]")
            else:
                print(f"✗ Error saving configuration: {e}")
            return None

    def run(self):
        try:
            config = self.generate_config()
            if config:
                # Preview config
                if self.console:
                    self.console.print("\n[bold]Configuration Preview:[/bold]")
                    import tomli_w
                    toml_str = tomli_w.dumps(config)
                    self.console.print(Panel(toml_str, title="Generated Config", border_style="green"))
                else:
                    print("\nConfiguration Preview:")
                    import tomli_w
                    print(tomli_w.dumps(config))
                
                if self.get_confirm("Save this configuration?", True):
                    self.save_config(config)
                    
                if self.get_confirm("Generate another configuration?", False):
                    self.run()
        except KeyboardInterrupt:
            if self.console:
                self.console.print("\n[yellow]Configuration cancelled.[/yellow]")
            else:
                print("\nConfiguration cancelled.")
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")

def main():
    """Entry point for model config TUI."""
    # Check for required dependencies
    try:
        import tomli_w
    except ImportError:
        print("Error: tomli-w is required for generating TOML files.")
        print("Install with: pip install tomli-w")
        sys.exit(1)
    
    if not HAS_RICH:
        print("Note: Install 'rich' for enhanced interface: pip install rich")
    
    tui = ModelConfigTUI()
    tui.run()

if __name__ == "__main__":
    main()