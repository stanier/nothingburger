#!/usr/bin/env python3
"""Simple validation for model configuration files."""

import tomllib
from pathlib import Path
from typing import List, Tuple

class ConfigValidator:
    """Simple validator for model configurations."""
    
    def __init__(self):
        self.required_fields = {
            'service': ['provider'],
        }
        
        self.provider_requirements = {
            'openai': ['base_url', 'model_key'],
            'ollama': ['base_url', 'model_key'], 
            'llama-cpp-python': ['model_path'],
        }

    def validate_file(self, filepath: str) -> Tuple[bool, List[str]]:
        """Validate a configuration file. Returns (is_valid, errors)."""
        errors = []
        
        # Check file exists
        if not Path(filepath).exists():
            return False, [f"File not found: {filepath}"]
        
        try:
            # Parse TOML
            with open(filepath, 'rb') as f:
                config = tomllib.load(f)
            
            # Basic validation
            if 'service' not in config:
                errors.append("Missing [service] section")
                return False, errors
            
            service = config['service']
            
            # Check provider
            if 'provider' not in service:
                errors.append("Missing service.provider")
                return False, errors
            
            provider = service['provider']
            if provider not in self.provider_requirements:
                errors.append(f"Unknown provider: {provider}")
                return False, errors
            
            # Check provider-specific requirements
            for field in self.provider_requirements[provider]:
                if field not in service:
                    errors.append(f"Missing required field: service.{field}")
            
            # Basic generation parameter validation
            if 'generation' in config:
                gen = config['generation']
                if 'temperature' in gen and not (0.0 <= gen['temperature'] <= 2.0):
                    errors.append("temperature must be between 0.0 and 2.0")
                if 'top_p' in gen and not (0.0 <= gen['top_p'] <= 1.0):
                    errors.append("top_p must be between 0.0 and 1.0")
                if 'max_tokens' in gen and gen['max_tokens'] < 1:
                    errors.append("max_tokens must be positive")
            
        except tomllib.TOMLDecodeError as e:
            errors.append(f"Invalid TOML: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return len(errors) == 0, errors

    def test_loading(self, filepath: str) -> Tuple[bool, str]:
        """Test if the configuration can be loaded."""
        try:
            from .model_loader import initializeModel
            model = initializeModel(filepath)
            return True, "Model loaded successfully"
        except Exception as e:
            return False, str(e)

def main():
    """CLI for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate model configurations")
    parser.add_argument('config_file', help='Configuration file to validate')
    parser.add_argument('--test-loading', action='store_true', help='Test model loading')
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    is_valid, errors = validator.validate_file(args.config_file)
    
    print(f"Validating: {args.config_file}")
    
    if is_valid:
        print("✓ Configuration is valid")
        
        if args.test_loading:
            success, message = validator.test_loading(args.config_file)
            if success:
                print("✓ Model loads successfully")
            else:
                print(f"✗ Model loading failed: {message}")
                return 1
    else:
        print("✗ Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())