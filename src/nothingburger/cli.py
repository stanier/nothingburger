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