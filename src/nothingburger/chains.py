import asyncio
from nothingburger.memory import ConversationalMemory

class Chain:
    template = None
    output_parser = None
    model = None
    debug = False
    stream = False

    def __init__(self, **kwargs):
        self.template = kwargs.get('template', self.template)
        self.output_parser = kwargs.get('output_parser', self.output_parser)
        self.model = kwargs.get('model', self.model)
        self.debug = kwargs.get('debug', self.debug)
        self.stream = kwargs.get('stream', self.stream)

    def generate(self, inp, **kwargs):
        prompt = self.template.render(inp=inp)
        if self.debug: print(prompt)
        result = self.model.generate(prompt, stream=self.stream, **kwargs)
        if self.debug: print(result)
        return result

class InstructChain(Chain):
    instruction = None
    response_prefix = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.instruction = kwargs.get('instruction', self.instruction)

    def generate(self, inp, **kwargs):
        prompt = self.template.render(
            inp=inp,
            instruction=kwargs.get('instruction', self.instruction),
            response_prefix=kwargs.get('response_prefix', self.response_prefix),
        )
        if self.debug: print(prompt)
        result = self.model.generate(prompt, stream=self.stream, **kwargs)
        if self.debug: print(result)
        return result

class ChatChain(InstructChain):
    system_prefix = ""
    system_suffix = ""
    assistant_prefix = ""
    assistant_suffix = ""
    user_prefix = ""
    user_suffix = ""
    memory = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prefix = kwargs.get('system_prefix', self.system_prefix)
        self.system_suffix = kwargs.get('system_suffix', self.system_suffix)
        self.assistant_prefix = kwargs.get('assistant_prefix', self.assistant_prefix)
        self.assistant_suffix = kwargs.get('assistant_suffix', self.assistant_suffix)
        self.user_prefix = kwargs.get('user_prefix', self.user_prefix)
        self.user_suffix = kwargs.get('user_suffix', self.user_suffix)
        self.memory = kwargs.get('memory', ConversationalMemory())
        self.stop = kwargs.get('stop', [])
        self.use_chat_format = getattr(self.model, 'api_format', 'completions') == 'chat'

    def generate(self, inp, **kwargs):
        template = kwargs.get('template', self.template)
        memory = kwargs.get('memory', self.memory)
        
        if self.use_chat_format:
            result = self.model.generate(inp, **kwargs)
        else:
            prompt = template.render(
                inp=inp,
                instruction=kwargs.get('instruction', self.instruction),
                system_prefix=kwargs.get('system_prefix', self.system_prefix),
                system_suffix=kwargs.get('system_suffix', self.system_suffix),
                assistant_prefix=kwargs.get('assistant_prefix', self.assistant_prefix),
                assistant_suffix=kwargs.get('assistant_suffix', self.assistant_suffix),
                user_prefix=kwargs.get('user_prefix', self.user_prefix),
                user_suffix=kwargs.get('user_suffix', self.user_suffix),
                memory=memory,
            )
            if self.debug: print(prompt)
            result = self.model.generate(prompt, **kwargs)
        
        memory.add_message(kwargs.get('user_prefix', self.user_prefix), inp)
        
        if self.debug and not self.stream: print(result)

        if self.stream:
            def stream_handler(result):
                message = ""
                for part in result:
                    piece = part.get('response', '') if isinstance(part, dict) else str(part)
                    message += piece
                memory.add_message(kwargs.get('assistant_prefix', self.assistant_prefix), message)
                return result
            return stream_handler(result)
        else:
            memory.add_message(kwargs.get('assistant_prefix', self.assistant_prefix), result)

        return result
    
    def run(self, inp, **kwargs):
        return self.generate(inp, **kwargs)