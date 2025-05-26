#!/usr/bin/env python3
"""
Command line interface for nothingburger.
"""

import argparse
import sys

class bcolors:
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'
    ITALICS     = '\033[3m'
    BELL        = '\007'
    DEBUG       = '\033[38;5;3m'

def repl(chain, user_identifier = "User", assistant_identifier = "Assistant"):
    while True:
        raw_input = input(f"{bcolors.BOLD}{user_identifier} >{bcolors.ENDC} ")
        if (raw_input == "quit" or raw_input == "exit"): break

        result = chain.generate(raw_input)

        print(f"{bcolors.BOLD}{assistant_identifier} >{bcolors.ENDC} ", end = "")
        if chain.stream:
            for part in result:
                print(part['response'], end='', flush=True)
            print('\n')
        else: 
            print(result)

def config_model():
    """Launch the model configuration TUI."""
    try:
        from .config_tui import ModelConfigTUI
        tui = ModelConfigTUI()
        tui.run()
    except ImportError as e:
        print(f"Error importing config TUI: {e}")
        print("Make sure tomli-w is installed: pip install tomli-w")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='nothingburger',
        description='Minimalistic framework for interfacing with LLMs'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Model configuration TUI
    config_parser = subparsers.add_parser(
        'config-model',
        help='Launch interactive model configuration generator'
    )
    
    # Chat command (basic example)
    chat_parser = subparsers.add_parser(
        'chat',
        help='Start an interactive chat session'
    )
    chat_parser.add_argument('--model-library', default='./.model_library', help='Path to model library')
    chat_parser.add_argument('--model-file', default='ollama/neural-chat.toml', help='Model configuration file')
    chat_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.command == 'config-model':
        config_model()
    elif args.command == 'chat':
        # Basic chat implementation
        try:
            from .model_loader import initializeModel
            from .chains import ChatChain
            from .memory import ConversationalMemory
            import nothingburger.templates as templates
            
            model = initializeModel(f"{args.model_library}/{args.model_file}")
            chain = ChatChain(
                model=model,
                template=templates.getTemplate("alpaca_instruct_chat"),
                debug=args.debug,
                memory=ConversationalMemory()
            )
            
            print(f"{bcolors.BOLD}Nothingburger Chat{bcolors.ENDC}")
            print("Type 'quit' or 'exit' to end the session.")
            repl(chain)
            
        except Exception as e:
            print(f"Error starting chat: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()