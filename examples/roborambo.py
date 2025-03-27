from datetime import datetime

import argparse
import sys
import json
import requests
import re
import ast
import time
import tomllib

import pandoc
import chromadb

from zulip import Client as ZulipClient
#from mattermostdriver import Driver as MattermostClient
#from rocketchat_API.rocketchat import RocketChat as RocketchatClient
#from discord import Client as DiscordClient
#from pymsteams import connectorcard as TeamsClient

from nothingburger.chains import ChatChain
from nothingburger.memory import ConversationalMemory
from nothingburger.cli import bcolors
from nothingburger.model_loader import initializeModel

import nothingburger.templates as templates

from pandoc.types import *

active_tools = {}

enabled_tools = [
    'web',
    'inspector',
]

enabled_clients = [
    "zulip",
]

privileged_users = {
    "zulip": [
        "fprefect@heartogold.local",
    ],
}

tunables = {
    'frequency_penalty': 1.07,
    'max_tokens': 1024,
    'presence_penalty': 0.0,
    'seed': 42,
    #'stop': '',
    'temperature': 0.0,
    'top_p': 1.0,
    'top_k': -1,
    #'typical_p': '',
    'mirostat': 0,
    'mirostat_eta': 0.1,
    'mirostat_tau': 5.0,
}

options = {
    "DEBUG"         : False,
    "PERSIST_MEMORY": True,
    "ZULIP_CONF"    : "~/.zuliprc.rambo",
    "MODEL_LIBRARY" : "./.model_library",
    "MODEL_FILE"    : "ollama/neural-chat.toml",
    "NAME"          : "Son of Rambo",
    "TEAM"          : "Hitchhikers",
    "SITE"          : "Heart o' Gold",
    "PERSONA"       : "You are {NAME}, an AI assistant powered by an LLM ran on-premises by {TEAM} at {SITE}.",
    "INSTRUCTION"   : "{PERSONA}\n\n{TOOL_INSTRUCTIONS}\n\n{SCENE_INSTRUCTIONS}\n\n{TIMESTAMP_INSTRUCTIONS}",
    "SCENE_INSTRUCTIONS"    : "You have access to an instant messaging service that enables communication between members of {TEAM}.  Continue the conversation history provided in Input",
    "TIMESTAMP_INSTRUCTIONS": "A timestamp will accompany each message, surrounded by brackets.  When referring to the time, do so in a natural human-readable way",
    "TOOL_INSTRUCTIONS": "You have access to the following tools:\n{TOOLS}To use a tool, respond with `INVOKE tool.function(arg_foo = \"lorem\", arg_bar = 42)` where the tool, function and arguments appropriately complement the tool you wish to use",
    "TOOL_ENTRY_TEMPLATE"     : "{tool_name}: {tool_desc}\n{func_entries}\n",
    "FUNC_ENTRY_TEMPLATE"     : "  - `{tool_slug}.{func_slug}`: {func_desc}\n    Args:{arg_entries}\n",
    "TOOL_ENTRY_TEMPLATE_NOEN": "{tool_name}: {tool_desc}\n",
    "FUNC_ENTRY_TEMPLATE_NOEN": "  - `{tool_slug}.{func_slug}`: {func_desc}\n",
    "ARGS_ENTRY_TEMPLATE"     : "      - `{arg_slug}` (`{arg_type}`): {arg_desc}",
    "CUTOFF_PHRASE"           : "bicycle built for two",
    "CUTOFF_MESSAGE"          : "Emergency cutoff activated.  {name} is now halted.",
    "CUTOFF_HINT"             : "It won't be a stylish marriage, I can't afford a carriage, But you'll look sweet upon the seat Of a [cutoff phrase]!",
    "EMOJI_LOOK"      : "eyes",
    "EMOJI_READ"      : "book",
    "EMOJI_WRITE"     : "pencil",
    "EMOJI_WORK"      : "working on it",
    "EMOJI_ASK"       : "umm",
    "EMOJI_POINT"     : "point up",
    "EMOJI_PLAN"      : "thought",
    "EMOJI_YES"       : "+1",
    "EMOJI_NO"        : "-1",
    "EMOJI_MAYBE"     : "palm hand down",
    "EMOJI_OCTO"      : "octopus", # 🐙
    "EMOJI_SUCCESS"   : "check",
    "EMOJI_FAILURE"   : "cross mark",
    "EMOJI_TOOL"      : "toolbox",
    "EMOJI_WEB"       : "globe",
    "EMOJI_SEARCH"    : "search",
    "EMOJI_NOACCESS"  : "prohibited",
    "IGNORING_MESSAGE": "Received message, but we are not mentioned and don't feel like responding",
}

class MemoryBackend:
    def __init__(self, **kwargs):
        pass

class SqlMemoryBackend(MemoryBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MemcacheMemoryBackend(MemoryBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Tool:
    name = "Unconfigured Tool"
    emoji = options["EMOJI_TOOL"]
    description = "This is a tool with no real metadata attached to it!  This is more likely than not unintended and should be brought to the attention of the operator immediately"

    def __init__(self, **kwargs):
        self.methods = {
            'foo': {
                'description': 'dummy func',
                'method': self.dummy,
                'arguments': {
                    'bar': {
                        'type': 'string',
                        'description': 'Lorem ipsum dolor sit amet',
                    }
                }
            },
        }
    
    def dummy(self, **kwargs):
        pass

    def get_methods(self, **kwargs):
        return self.methods

class InspectorTool(Tool):
    name = "Tool Inspector"
    description = "Allows you to gather further insight into tools."
    emoji = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'inspect': {
                'description': 'Get more information about a given tool, including available functions and their arguments.  (Hint: `inspector.inspect(tool_slug = "web")`)',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.inspect,
                'arguments': {
                    'tool_slug': {
                        'type': 'string',
                        'description': 'The slug used to refer to the tool that should be described.',
                    },
                }
            },
            #'describe': {
            #    'description': '',
            #    'method': self.describe,
            #    'arguments': {
            #        'tool_slug': {
            #            'type': 'string',
            #            'description': 'The slug used to refer to the tool that should be described',
            #        }
            #    }
            #},
        }

    def inspect(self, **kwargs):
        target_tool = active_tools[kwargs.get('tool_slug', 'inspector')]

        funcs_concat = ""
        for func in target_tool['functions']: # functions are required
            args_concat = ""
            for arg in target_tool['functions'][func].get('arguments', {}): # arguments are optional
                args_concat = "{}\n{}".format(args_concat, options['ARGS_ENTRY_TEMPLATE'].format(
                    arg_slug = arg,
                    arg_type = target_tool['functions'][func]['arguments'][arg]['type'],
                    arg_desc = target_tool['functions'][func]['arguments'][arg]['description'],
                ))
            funcs_concat = "{}{}".format(funcs_concat, options['FUNC_ENTRY_TEMPLATE'].format(
                func_slug = func,
                func_desc = target_tool['functions'][func]['description'],
                tool_slug = tool,
                arg_entries = args_concat,
            ))

        tool_string = "```\n{}```".format(options['TOOL_ENTRY_TEMPLATE'].format(
            tool_name = target_tool['name'],
            tool_desc = target_tool['description'],
            func_entries = funcs_concat,
        ))

        return tool_string

    def describe(self, **kwargs):
        pass

class WebTool(Tool):
    name = "Web Engine"
    emoji = options["EMOJI_WEB"]
    description = "Enables you to search and navigate the web"

    search_endpoint = 'https://stract.com/beta/api/search'

    headers = {
        'Content-type': 'application/json'
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'search': {
                'description': 'Search the web',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.search,
                'arguments': {
                    'query': {
                        'type': 'string',
                        'description': 'Query to pass to the web search engine',
                    },
                }
            },
            'render': {
                'description': 'Render a webpage',
                'method': self.render,
                'arguments': {
                    'site_uri': {
                        'type': 'string',
                        'description': 'URL of the webpage that should be rendered',
                    },
                    'skip_certs': {
                        'type': 'boolean',
                        'description': 'Whether or not we should accept expired TLS certificates',
                    },
                }
            },
            'read': {
                'description': 'Read the text content of a webpage',
                'method': self.read,
                'arguments': {
                    'site_uri': {
                        'type': 'string',
                        'description': 'URL of the webpage that should be rendered',
                    },
                    'skip_certs': {
                        'type': 'boolean',
                        'description': 'Whether or not we should accept expired TLS certificates',
                    },
                }
            },
            'download': {
                'description': 'Download a file from the internet',
                'method': self.download,
                'arguments': {
                    'file_uri': {
                        'type': 'string',
                        'description': 'URL we should download a file from',
                    },
                    'file_name': {
                        'type': 'string',
                        'description': 'Name the downloaded file should be saved as'
                    },
                }
            },
        }

    def search(self, query, **kwargs):
        # TODO:  Support multiple search providers
        data = json.dumps({"query": query})

        response = requests.post(self.search_endpoint, headers = self.headers, data = data).json()

        results = []

        for webpage in response["webpages"]:
            content = ""

            for fragment in webpage['snippet']['text']['fragments']:
                content += fragment['text']

            results.append({"title": webpage['title'], "snippet": content})

        return results

    def render(self, **kwargs):
        # TODO:  Render the page elements (CSS, optionally JS, etc) maybe with Selenium?
        pass

    def read(self, site_uri, **kwargs):
        response = requests.get(site_uri, headers = self.headers)

        doc = pandoc.read(response.text, format = "html")
        md = f'```{pandoc.write(doc, format = "markdown")}```'

        return md

    def download(self, **kwargs):
        pass

class FileTool(Tool):
    name = "File Browser"
    description = "Allows you to search for, read and write files locally"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'search': {
                'description': 'Search for a given filename',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.search,
                'arguments': {
                    'scope': {
                        'type': 'string',
                        'description': 'Pathname of filesystem directory the search should occur within',
                    },
                    'query': {
                        'type': 'string',
                        'description': 'Part of the filename being searched for',
                    },
                }
            },
            'read': {
                'description': 'Read a given filename',
                'method': self.read,
                'arguments': {
                    'filename': {
                        'type': 'string',
                        'description': 'Pathname of the file to read',
                    },
                }
            },
            'write': {
                'description': 'Write content to a given filepath',
                'method': self.write,
                'arguments': {
                    'filename': {
                        'type': 'string',
                        'description': 'Pathname of the file to write',
                    },
                }
            },
        }

    def search(self, **kwargs):
        pass

    def read(self, **kwargs):
        pass

    def write(self, **kwargs):
        pass

class ChatTool(Tool):
    name = "Chat Interface"
    description = "Allows you to interact with the instant messaging system to perform actions and search through conversation histories"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'search': {
                'description': 'Search the conversation history of public message channels',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.search,
                'arguments': {
                    'query': {
                        'type': 'string',
                        'description': 'Phrase to search for in the chat server history',
                    },
                }
            },
            'message': {
                'description': 'Send a message to a given user',
                'method': self.message,
                'arguments': {
                    'user_name': {
                        'type': 'string',
                        'description': 'Name of the user to deliver the message to',
                    },
                    'message': {
                        'type': 'string',
                        'description': 'Message to deliver to the user',
                    },
                }
            },
        }

    def search(self, **kwargs):
        pass

    def message(self, **kwargs):
        pass

class KnowledgebaseTool(Tool):
    name = "Knowledgebase Explorer"
    description = "Allows you to explore an internal knowledge base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'search': {
                'description': 'Search the internal knowledge base',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.search,
                'arguments': {
                    'query': {
                        'type': 'string',
                        'description': 'Phrase to search the knowledgebase for',
                    },
                }
            },
        }

    def search(self, **kwargs):
        pass

class GraphQLTool(Tool):
    name = "GraphQL"
    description = "Enables you to submit read-only GraphQL queries to abritrary endpoints"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'query': {
                'description': 'Query a GraphQL endpoint',
                'method': self.query,
                'arguments': {
                    'endpoint': {
                        'type': 'string',
                        'description': 'URI of the endpoint we should talk to',
                    },
                    'query': {
                        'type': 'string',
                        'description': 'GraphQL query to send to the endpoint',
                    },
                }
            },
            'readdocs': {
                'description': 'Get more information about a specific GraphQL endpoint',
                'method': self.readdocs,
                'arguments': {
                    'docs_path': {
                        'type': 'string',
                        'description': 'URI to the page hosting the endpoint\'s documentation',
                    },
                }
            },
        }

    def query(self, **kwargs):
        pass

    def readdocs(self, **kwargs):
        pass

class VectorStoreTool(Tool):
    name = "Vector Store"
    description = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.client = chromadb.Client()

        self.methods = {
            'search': {
                'description': 'Search the vector store for a given query',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.search,
                'arguments': {
                    'query': {
                        'type': 'string',
                        'description': 'bar',
                    },
                }
            },
            'embed_note': {
                'description': 'Embed a piece of information into the vector store',
                'method': self.embed_note,
                'arguments': {
                    'content': {
                        'type': 'string',
                        'description': 'Text content that should be embedded in the vector store',
                    },
                    'results': {
                        'type': 'integer',
                        'description': 'Number of results to give back',
                    },
                }
            },
        }
    
    def search(self, **kwargs):
        results = collection.query(
            query_texts = [kwargs['query']],
            n_results = kwargs['results'],
        )

        return results

    def embed_note(self, **kwargs):
        collection.add(
            documents = [kwargs['content']],
            metadatas = [{"source": options["NAME"], "timestamp": datetime.now()}],
            ids = [f"note{int(time.time())}"],
        )

class ExpertAskTool(Tool):
    name = "Ask An Expert"
    description = "Allows you to ask an expert for input on a question or topic, in natural language"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'ask': {
                'description': 'Ask an expert',
                'emoji': options["EMOJI_ASK"],
                'method': self.ask,
                'arguments': {
                    'query': {
                        'type': 'string',
                        'description': 'Question to ask',
                    }
                }
            },
        }

    def ask(self, **kwargs):
        pass

class InternalScheduleTool(Tool):
    name = "Schedule Tool (internal)"
    description = "A tool that allows you to create and check schedules, including leaving notes about things that should be done at specific times"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.methods = {
            'find_time': {
                'description': 'Look for time in a schedule to dedicate to something',
                'method': self.find_time,
                'arguments': {
                    'window': {
                        'type': 'string',
                        'description': 'Timeframe that the allocated timeslot should fall within',
                    },
                    'duration': {
                        'type': 'string',
                        'description': 'How much time needs to be set aside',
                    },
                }
            },
            'lookup': {
                'description': 'Look for something in the schedule',
                'emoji': options["EMOJI_SEARCH"],
                'method': self.lookup,
                'arguments': {
                    'foo': {
                        'type': 'string',
                        'description': 'Query to lookup',
                    }
                }
            },
        }

    def find_time(self, **kwargs):
        pass

    def lookup(self, **kwargs):
        pass

    def add(self, **kwargs):
        pass

    def remove(self, **kwargs):
        pass

available_tools = {
    'inspector':        InspectorTool,
    'web':              WebTool,
    'file':             FileTool,
    'chat':             ChatTool,
    'knowledgebase':    KnowledgebaseTool,
    'graphql':          GraphQLTool,
    'vectorstore':      VectorStoreTool,
    'expert':           ExpertAskTool,
    'schedule':         InternalScheduleTool,
}

tools_concat = ""
for tool in enabled_tools:
    active_tools[tool] = available_tools[tool]()

    funcs_concat = ""
    for func in active_tools[tool].methods: # functions are required
        args_concat = ""
        for arg in active_tools[tool].methods[func].get('arguments', {}): # arguments are optional
            args_concat = "{}\n{}".format(args_concat, options['ARGS_ENTRY_TEMPLATE'].format(
                arg_slug = arg,
                arg_type = active_tools[tool].methods[func]['arguments'][arg]['type'],
                arg_desc = active_tools[tool].methods[func]['arguments'][arg]['description'],
            ))
        funcs_concat = "{}{}".format(funcs_concat, options['FUNC_ENTRY_TEMPLATE_NOEN'].format(
            func_slug = func,
            func_desc = active_tools[tool].methods[func]['description'],
            tool_slug = tool,
            #arg_entries = args_concat,
            arg_entries = "",
        ))
    tools_concat = "{}{}".format(tools_concat, options['TOOL_ENTRY_TEMPLATE'].format(
        tool_name = active_tools[tool].name,
        tool_desc = active_tools[tool].description,
        func_entries = funcs_concat,
        #func_entries = "",
    ))

settings = {}
settings['SITE']                    = options['SITE'].format(**settings)
settings['TEAM']                    = options['TEAM'].format(**settings)
settings['NAME']                    = options['NAME'].format(**settings)
settings['PERSONA']                 = options['PERSONA'].format(**settings)
settings['TOOLS']                   = tools_concat
settings['TOOL_INSTRUCTIONS']       = options['TOOL_INSTRUCTIONS'].format(**settings)
settings['TIMESTAMP_INSTRUCTIONS']  = options['TIMESTAMP_INSTRUCTIONS'].format(**settings)
settings['SCENE_INSTRUCTIONS']      = options['SCENE_INSTRUCTIONS'].format(**settings)
settings['INSTRUCTION']             = options['INSTRUCTION'].format(**settings)
settings['CUTOFF_PHRASE']           = options['CUTOFF_PHRASE'].format(**settings)
settings['CUTOFF_HINT']             = options['CUTOFF_HINT'].format(**settings)

def parse_invocation(invocation, **kwargs):
    if "INVOKE" not in invocation[0:6]: return False
    
    match = re.search(r"^INVOKE\s(\w+)\.(\w+)\((.+)\)", invocation)

    tool_args = {}
    for arg in re.findall(r"(?i)(\w+)\s?\=\s?(?:((?:true)|(?:false))|('[^'\|\n)]+')|(\"[^\"\|\n)]+\")|(\[.*\])|(\{.*\})|(\d+.\d+)|(\w+))?", match.group(3)):
        if arg[1]: # Boolean
            value = (arg[1].lower() == 'true')
        if arg[2] or arg[3]: # String
            value = (arg[2] + arg[3])[1:-1]
        if arg[4]: # Array/list
            value = arg[4][1:-1]
        if arg[5]: # Object
            value = arg[5][1:-1]
        if arg[6]: # Float
            value = float(arg[6])
        if arg[7] and arg[7].isnumeric(): # Int
            value = int(arg[7])

        print(f"ARG {arg[0]} = {value}")
        tool_args[arg[0]] = value
    
    return { 'tool': match.group(1), 'func': match.group(2), 'args': tool_args }

class RamboChain(ChatChain):
    # TODO:  So yeah, this should be an actual DB of some sort.... but we're going by prototype principles atm so it's fiiiiine
    memory_db = {}
    
    def __init__(self, **kwargs): super().__init__(**kwargs)

    def responsiveness_simple(self, message, assistant_prefix, **kwargs):
        assessment = self.generate(
            message,
            instruction = f"Given the message in Input sent by a user, determine whether the assistant \"{assistant_prefix}\" should read it and indicate this with either a Yes or No.  The Assistant should read the message if it is addressed to.  If they mention they don't want their message read by the assistant, it shouldn't read it",
            template    = templates.getTemplate("alpaca_instruct_input"),
            max_tokens  = 100, # Should be 1, but Ollama currently bugs with low num_predict
            memory      = None,
            top_k       = -1,
            top_p       = 1.0,
            **kwargs,
        )
        return assessment[0] == "Y"

    def cutoff(self, msg, **kwargs): return settings['CUTOFF_PHRASE'].replace(" ", "") in msg.upper().replace(" ", "")
    def step(self, sender, content, **kwargs):
        convmem = kwargs.get('memory', {})
        
        response = self.generate(
            content,
            user_prefix = sender,
            **dict(kwargs, **tunables),
        )

        # Add input message from user to memory
        convmem.add_message(
            role = sender,
            content = content,
            timestamp = kwargs.get('timestamp', datetime.now()),
        )

        # Add assistant's response to memory
        convmem.add_message(
            role = kwargs['assistant_prefix'],
            content = response,
            timestamp = datetime.now(),
        )

        return response

    def run(self, message, callbacks, **kwargs):
        convkey = ""

        if self.cutoff(message['content']) is True:
            callbacks["cutoff"](message)
            return

        sender = message['sender'] # Message sent by
        recips = message['recips'] # Message addressed to
        source = message['source'] # Messaging client received from
        content = message['content'] # The main content of the message
        channel = message['channel'] # Channel the message was sent to
        server = message['server'] # Server the message was sent to
        visibility = message['visibility'] # The general visibility of the message
        privacy = message['privacy'] # How private the message is
        secure = message['secure'] # Whether the message received over a secure, end-to-end encrypted channel

        if privacy == 'private_direct':
            pass
        elif privacy == 'private_group':
            if not self.responsiveness_simple(content, self.assistant_prefix):
                if options["DEBUG"]: sys.stdout.write(f"{options['IGNORED_MESSAGE']}\n")
                return
        else:
            if not self.responsiveness_simple(content, self.assistant_prefix):
                if options["DEBUG"]: sys.stdout.write(f"{options['IGNORED_MESSAGE']}\n")
                return

        convhash = hash(convkey)
        if convhash not in self.memory_db: self.memory_db[convhash] = ConversationalMemory()
        convmem = self.memory_db[convhash]

        callbacks["start"](message)

        response = self.step(
            sender['name'],
            content,
            memory = convmem,
            **kwargs,
        )
        
        invocation = parse_invocation(response)
        while invocation:
            callbacks["tool"](message, invocation)
            result = active_tools[invocation['tool']].methods[invocation['func']]['method'](**invocation['args'])

            response = self.step(
                f"{invocation['tool']}.{invocation['func']}",
                result,
                memory = convmem,
                **kwargs,
            )

            invocation = parse_invocation(response)

        callbacks["finish"](message)
        return response

class MessagingInterface:
    def __init__(self, chain, **kwargs):
        pass
    
    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

class ZulipInterface(MessagingInterface):
    consolecolor = (40, 177, 249)
    consolename = "Zulip"
    sourcename = "zulip"

    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)
        self.client = ZulipClient(config_file = kwargs['config_file'])

        self.chain = chain
        self.profile = self.client.get_profile()
        
        self.client.call_on_each_message(self.handle_message)

    def start_callback(self, message, **kwargs):
        self.add_reaction(message['id'], options['EMOJI_LOOK'])

    def tool_callback(self, message, invocation, **kwargs):
        if active_tools[invocation['tool']].emoji:
            self.add_reaction(message['id'], active_tools[invocation['tool']].emoji)
        if active_tools[invocation['tool']].methods[invocation['func']].get('emoji', False):
            self.add_reaction(message['id'], active_tools[invocation['tool']].methods[invocation['func']]['emoji'])

    def finish_callback(self, message, **kwargs):
        self.remove_reaction(message['id'], options['EMOJI_LOOK'])
        self.remove_reaction(message['id'], options['EMOJI_WRITE'])

    def write_callback(self, message, **kwargs):
        self.add_reaction(message['id'], options['EMOJI_WRITE'])

    def cutoff_callback(self, message, **kwargs):
        if message['type'] == 'private':
            recips = []
            for r in message['display_recipient']: recips.append(r['id'])
            msg_type = "private"
            msg_to = recips
        else:
            msg_type = "stream",
            msg_to = message['stream_id']
        
        self.client.send_message({"type": msg_type, "to": msg_to, "content": CUTOFF_MESSAGE})
        sys.exit()

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, mid, emoji): self.client.add_reaction({"message_id": mid, "emoji_name": emoji})
    def remove_reaction(self, mid, emoji): self.client.remove_reaction({"message_id": mid, "emoji_name": emoji})

    def handle_message(self, message, **kwargs):
        if message['sender_id'] == self.profile['user_id']: return # Don't talk to yourself

        recips = []
        ri = []
        rs = []
        for recip in message['display_recipient']:
            ri.append(int(recip['id']))
            rs.append(str(recip['id']))
            recips.append({
                'id': recip['id'],
                'full_name': recip['full_name'],
                'email': recip['email'],
            })
        ri.sort()

        if message['type'] == 'private':
            visibility = 'private'
            if len(recips) > 2:
                privacy = 'private_group'
            else:
                privacy = 'private_direct'

            channel = ','.join(rs)
            to = ri # Don't think we can use channel because Zulip API expects array?  I think?  I should double check
        else:
            visibility = 'semipublic'
            privacy = 'semipublic'
            channel = message['stream_id']
            to = channel

        if message['content'][:4] == "TUNE":
            if message['sender_email'] not in privileged_users['zulip']:
                self.client.add_reaction({"message_id": message['id'], "emoji_name": options['EMOJI_NOACCESS']})
                return
            
            tune_args = {}
            for arg in re.findall(r"(?i)(\w+)\s?\=\s?(?:((?:true)|(?:false))|('[^'\|\n)]+')|(\"[^\"\|\n)]+\")|(\[.*\])|(\{.*\})|(\d+.\d+)|(\w+))?", message['content'][5:]):
                if arg[1]: # Boolean
                    value = (arg[1].lower() == 'true')
                if arg[2] or arg[3]: # String
                    value = (arg[2] + arg[3])[1:-1]
                if arg[4]: # Array/list
                    value = arg[4][1:-1]
                if arg[5]: # Object
                    value = arg[5][1:-1]
                if arg[6]: # Float
                    value = float(arg[6])
                if arg[7] and arg[7].isnumeric(): # Int
                    value = int(arg[7])

                tune_args[arg[0]] = value
            
            tunables.update(tune_args)
            self.client.add_reaction({"message_id": message['id'], "emoji_name": options['EMOJI_SUCCESS']})
            return

        if message['content'][:8] == "TUNABLES":
            if message['sender_email'] not in privileged_users:
                self.client.add_reaction({"message_id": message['id'], "emoji_name": options['EMOJI_NOACCESS']})
                return
            
            self.client.send_message({"type": message['type'], "to": to, "content": f"The following tunables are set globally:\n```\n{tunables}\n```"})
            return

        msg = {
            'id': message['id'],
            'sender': {
                'name': message['sender_full_name'],
                'email': message['sender_email'],
                'id': message['sender_id'],
            },
            'recips': recips,
            'source': self.sourcename,
            'content': message['content'],
            'channel': channel,
            'server': 'default',
            'visibility': visibility,
            'privacy': privacy,
            'secure': False,
        }

        response = self.chain.run(
            msg,
            callbacks = {
                'start'         : self.start_callback,
                'finish'        : self.finish_callback,
                'write'         : self.write_callback,
                'cutoff'        : self.cutoff_callback,
                'success'       : self.success_callback,
                'failure'       : self.failure_callback,
                'warning'       : self.warning_callback,
                'info'          : self.info_callback,
                'intervention'  : self.intervention_callback,
                'tool'          : self.tool_callback,
            },
            assistant_prefix = self.profile['full_name'],
            stop = ["\n[", "</s>"],
        )
        
        self.client.send_message({"type": message['type'], "to": to, "content": response})

class TeamsInterface(MessagingInterface):
    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)

    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

class MattermostInterface(MessagingInterface):
    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)

        self.client = MattermostClient({
            'url': '',
            'login_id': '',
            'password': '',
            'token': '',
            'scheme': 'https',
            'port': 8065,
            'basepath': '/api/v4',
            'mfa_token': '',
            'auth': None,
            'timeout': 30,
            'request_timeout': None,
            'keepalive': False,
            'keepalive_delay': 5,
            'websocket_kw_args': None,
            'debug': False,
        })
    
    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

class MatrixInterface(MessagingInterface):
    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)

    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

class DiscordInterface(MessagingInterface):
    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)
    
    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

class RocketchatInterface(MessagingInterface):
    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)
        self.client = RocketchatClient('user', 'pass', server_url='https://demo.rocket.chat')
    
    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

class GoogleChatInterface(MessagingInterface):
    def __init__(self, chain, **kwargs):
        super().__init__(chain, **kwargs)
    
    def start_callback(self, message, **kwargs):
        pass

    def tool_callback(self, message, **kwargs):
        pass

    def finish_callback(self, message, **kwargs):
        pass

    def write_callback(self, message, **kwargs):
        pass

    def cutoff_callback(self, message, **kwargs):
        pass

    def success_callback(self, message, **kwargs):
        pass

    def failure_callback(self, message, **kwargs):
        pass

    def warning_callback(self, message, **kwargs):
        pass

    def info_callback(self, message, **kwargs):
        pass

    def intervention_callback(self, message, **kwargs):
        pass

    def reply_message(self, message, data, **kwargs):
        pass

    def send_message(self, destination, data, **kwargs):
        pass

    def add_reaction(self, message, data, **kwargs):
        pass

    def remove_reaction(self, message, data, **kwargs):
        pass

    def handle_message(self, message, **kwargs):
        pass

available_clients = {
    'zulip': ZulipInterface,
    #'teams': TeamsInferface,
    #'mattermost': MattermostInterface,
    #'matrix': MatrixInterface,
    #'discord': DiscordInterface,
    #'rocketchat': RocketchatInterface,
    #'gsuite': GoogleChatInferface,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action = 'store_true', help = 'Enable debugging mode', default = options['DEBUG'])
    #parser.add_argument('--nomem', action = 'store_true', help = 'Turn off persistent memory', default = options['PERSIST_MEMORY'])
    #parser.add_argument('--supervised', action = 'store_true', help = 'Disable all self-guided capabilites', default = options['SUPERVISED'])

    args = parser.parse_args()

    chain = RamboChain(
        model               = initializeModel("{}/{}".format(options['MODEL_LIBRARY'], options['MODEL_FILE'])),
        instruction         = settings['INSTRUCTION'],
        template            = templates.getTemplate("alpaca_instruct_chat"),
        debug               = args.debug,
        stream              = False,
        assistant_prefix    = settings['NAME'],
    )

    conf = {
        'zulip': {
            'config_file': options['ZULIP_CONF'],
        }
    }

    clients = {}
    for client in enabled_clients:
        clients[client] = available_clients[client](chain=chain, **conf[client])
        clients[client].pname = f"\x1b[38;2;{clients[client].consolecolor[0]};{clients[client].consolecolor[0]};{clients[client].consolecolor[0]}m{clients[client].consolename}"
        print(f"Messaging client {clients[client].pname} has been initialized")
