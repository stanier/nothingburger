import asyncio

from nothingburger.memory import ConversationalMemory

class Chain:
    template        = None
    output_parser   = None
    model           = None
    debug           = False
    stream          = False

    def __init__(self, **kwargs):
        self.template       = kwargs.get('template'     , self.template     )
        self.output_parser  = kwargs.get('output_parser', self.output_parser)
        self.model          = kwargs.get('model'        , self.model        )
        self.debug          = kwargs.get('debug'        , self.debug        )
        self.stream         = kwargs.get('stream'       , self.stream       )

    def generate(self, inp, **kwargs):
        prompt = self.template.render(inp = inp)
        if self.debug: print(prompt)

        result = self.model.generate(prompt, stream = self.stream, **kwargs)
        if self.debug: print(result)
        return result

class InstructChain(Chain):
    instruction     = None
    response_prefix = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.instruction        = kwargs.get('instruction'      , self.instruction      )

    def generate(self, inp, **kwargs):
        prompt = self.template.render(
            inp = inp,
            instruction         = kwargs.get('instruction'      , self.instruction      ),
            response_prefix     = kwargs.get('response_prefix'  , self.response_prefix  ),
        )
        if self.debug: print(prompt)

        result = self.model.generate(prompt, stream = self.stream, **kwargs)
        if self.debug: print(result)
        return result

class ChatChain(InstructChain):
    system_prefix       = ""
    system_suffix       = ""
    assistant_prefix    = ""
    assistant_suffix    = ""
    user_prefix         = ""
    user_suffix         = ""
    memory              = None

    def __init__(self, **kwargs):
        print(f"💬 Initializing ChatChain...")
        print(f"  Model provided: {type(kwargs.get('model', 'None'))}")
        print(f"  Template provided: {type(kwargs.get('template', 'None'))}")
        print(f"  Instruction provided: {bool(kwargs.get('instruction'))}")
        
        super().__init__(**kwargs)
        self.system_prefix      = kwargs.get('system_prefix'    , self.system_prefix    )
        self.system_suffix      = kwargs.get('system_suffix'    , self.system_suffix    )
        self.assistant_prefix   = kwargs.get('assistant_prefix' , self.assistant_prefix )
        self.assistant_suffix   = kwargs.get('assistant_suffix' , self.assistant_suffix )
        self.user_prefix        = kwargs.get('user_prefix'      , self.user_prefix      )
        self.user_suffix        = kwargs.get('user_suffix'      , self.user_suffix      )
        self.memory             = kwargs.get('memory'           , ConversationalMemory())
        self.stop               = kwargs.get('stop'             , []                    )
        
        # Check if the model supports modern chat format
        self.use_chat_format = getattr(self.model, 'api_format', 'completions') == 'chat'
        
        print(f"  Model: {type(self.model)}")
        print(f"  Model api_format: {getattr(self.model, 'api_format', 'not specified')}")
        print(f"  Use chat format: {self.use_chat_format}")
        print(f"  Template: {self.template}")
        print(f"  Assistant prefix: '{self.assistant_prefix}'")
        print(f"  Memory: {type(self.memory)}")
        print(f"✅ ChatChain initialization complete!")

    def generate(self, inp, **kwargs):
        print(f"🔄 ChatChain.generate() called")
        print(f"  Input: {inp[:100]}{'...' if len(inp) > 100 else ''}")
        print(f"  Model: {type(self.model)}")
        print(f"  Use chat format: {self.use_chat_format}")
        print(f"  Kwargs keys: {list(kwargs.keys())}")
        
        template = kwargs.get('template', self.template)
        memory = kwargs.get('memory', self.memory)
        
        print(f"  Template: {type(template)}")
        print(f"  Memory: {type(memory)} with {len(memory.messages) if hasattr(memory, 'messages') else 0} messages")
        
        if self.use_chat_format:
            print("  🗨️  Using chat format - passing to model directly")
            # For chat format, just pass everything through to the model
            # The model adapter will handle memory and instruction formatting
            try:
                result = self.model.generate(inp, **kwargs)
                print(f"  ✅ Model generate successful, result length: {len(str(result))}")
            except Exception as e:
                print(f"  ❌ Model generate failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print("  📝 Using legacy format - rendering template")
            # Legacy format - use templates to create formatted prompt
            try:
                prompt = template.render(
                    inp                 = inp,
                    instruction         = kwargs.get('instruction'      , self.instruction      ),
                    system_prefix       = kwargs.get('system_prefix'    , self.system_prefix    ),
                    system_suffix       = kwargs.get('system_suffix'    , self.system_suffix    ),
                    assistant_prefix    = kwargs.get('assistant_prefix' , self.assistant_prefix ),
                    assistant_suffix    = kwargs.get('assistant_suffix' , self.assistant_suffix ),
                    user_prefix         = kwargs.get('user_prefix'      , self.user_prefix      ),
                    user_suffix         = kwargs.get('user_suffix'      , self.user_suffix      ),
                    memory              = memory,
                )
                print(f"  📝 Template rendered, prompt length: {len(prompt)}")
                if self.debug: print(prompt)
                
                result = self.model.generate(prompt, **kwargs)
                print(f"  ✅ Model generate successful, result length: {len(str(result))}")
            except Exception as e:
                print(f"  ❌ Template rendering or model generate failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Update memory
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

        print(f"🔄 ChatChain.generate() completed successfully")
        return result
    
    def run(self, inp, **kwargs):
        return self.generate(inp, **kwargs)