# src/nothingburger/config_validator.py
"""
Validation utilities for nothingburger model configurations.
"""

import os
import tomllib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ConfigValidator:
    """Validates nothingburger model configuration files."""
    
    def __init__(self):
        self.required_fields = {
            'service': ['provider'],
            'generation': []  # All generation fields are optional
        }
        
        self.provider_requirements = {
            'openai': {
                'service': ['base_url', 'model_key'],
                'optional': ['api_key', 'api_format']
            },
            'ollama': {
                'service': ['base_url', 'model_key'],
                'optional': ['api_format']
            },
            'llama-cpp-python': {
                'service': ['model_path'],
                'optional': ['n_gpu_layers', 'n_ctx', 'main_gpu']
            },
            'hf_transformers': {
                'service': ['model_key'],
                'optional': []
            },
            'ctransformers': {
                'service': ['model_path', 'model_type'],
                'optional': []
            }
        }
        
        self.valid_api_formats = ['chat', 'completions']

    def validate_file(self, filepath: str) -> Tuple[bool, List[str]]:
        """Validate a configuration file. Returns (is_valid, errors)."""
        errors = []
        
        try:
            # Check file exists and is readable
            if not Path(filepath).exists():
                errors.append(f"Configuration file not found: {filepath}")
                return False, errors
            
            # Parse TOML
            with open(filepath, 'rb') as f:
                config = tomllib.load(f)
            
            # Validate structure
            structure_errors = self._validate_structure(config)
            errors.extend(structure_errors)
            
            # Validate provider-specific requirements
            if 'service' in config and 'provider' in config['service']:
                provider_errors = self._validate_provider(config)
                errors.extend(provider_errors)
            
            # Validate generation settings
            generation_errors = self._validate_generation(config)
            errors.extend(generation_errors)
            
        except tomllib.TOMLDecodeError as e:
            errors.append(f"Invalid TOML syntax: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return len(errors) == 0, errors

    def _validate_structure(self, config: Dict) -> List[str]:
        """Validate basic configuration structure."""
        errors = []
        
        # Check required top-level sections
        if 'service' not in config:
            errors.append("Missing required section: [service]")
        
        # Check required service fields
        if 'service' in config:
            service = config['service']
            for field in self.required_fields['service']:
                if field not in service:
                    errors.append(f"Missing required field: service.{field}")
        
        return errors

    def _validate_provider(self, config: Dict) -> List[str]:
        """Validate provider-specific configuration."""
        errors = []
        
        provider = config['service']['provider']
        
        if provider not in self.provider_requirements:
            errors.append(f"Unknown provider: {provider}")
            return errors
        
        requirements = self.provider_requirements[provider]
        service = config['service']
        
        # Check required fields
        for field in requirements['service']:
            if field not in service:
                errors.append(f"Missing required field for {provider}: service.{field}")
        
        # Validate API format if present
        if 'api_format' in service:
            if service['api_format'] not in self.valid_api_formats:
                errors.append(f"Invalid api_format: {service['api_format']}. Must be one of: {self.valid_api_formats}")
        
        # Provider-specific validations
        if provider == 'openai':
            if 'base_url' in service and not service['base_url'].startswith(('http://', 'https://')):
                errors.append("OpenAI base_url must be a valid HTTP/HTTPS URL")
        
        elif provider == 'ollama':
            if 'base_url' in service and not service['base_url'].startswith(('http://', 'https://')):
                errors.append("Ollama base_url must be a valid HTTP/HTTPS URL")
        
        elif provider in ['llama-cpp-python', 'ctransformers']:
            if 'model_path' in service:
                model_path = Path(service['model_path'])
                # Only check if it's an absolute path or looks like a real file
                if model_path.is_absolute() and not model_path.exists():
                    errors.append(f"Model file not found: {service['model_path']}")
        
        return errors

    def _validate_generation(self, config: Dict) -> List[str]:
        """Validate generation parameters."""
        errors = []
        
        if 'generation' not in config:
            return errors
        
        generation = config['generation']
        
        # Validate numeric ranges
        numeric_validations = {
            'temperature': (0.0, 2.0),
            'top_p': (0.0, 1.0),
            'top_k': (-1, None),  # -1 or positive
            'max_tokens': (1, None),
            'presence_penalty': (-2.0, 2.0),
            'frequency_penalty': (-2.0, 2.0),
        }
        
        for field, (min_val, max_val) in numeric_validations.items():
            if field in generation:
                value = generation[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"generation.{field} must be a number")
                    continue
                
                if min_val is not None and value < min_val:
                    errors.append(f"generation.{field} must be >= {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"generation.{field} must be <= {max_val}")
        
        # Validate mirostat settings
        if 'mirostat' in generation:
            mirostat = generation['mirostat']
            if 'mode' in mirostat:
                if mirostat['mode'] not in [0, 1, 2]:
                    errors.append("generation.mirostat.mode must be 0, 1, or 2")
        
        return errors

    def test_model_loading(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """Test if the model configuration can be loaded. Returns (success, error)."""
        try:
            from .model_loader import initializeModel
            model = initializeModel(filepath)
            return True, None
        except Exception as e:
            return False, str(e)

def validate_config_cli():
    """Command-line interface for config validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate nothingburger model configurations")
    parser.add_argument('config_file', help='Path to configuration file')
    parser.add_argument('--test-loading', action='store_true', help='Test model loading')
    parser.add_argument('--quiet', action='store_true', help='Only output errors')
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    is_valid, errors = validator.validate_file(args.config_file)
    
    if not args.quiet:
        print(f"Validating: {args.config_file}")
    
    if is_valid:
        if not args.quiet:
            print("✓ Configuration is valid")
        
        if args.test_loading:
            success, error = validator.test_model_loading(args.config_file)
            if success:
                if not args.quiet:
                    print("✓ Model loads successfully")
            else:
                print(f"✗ Model loading failed: {error}")
                return 1
    else:
        print("✗ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(validate_config_cli())