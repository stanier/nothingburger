from jinja2 import Environment, DictLoader, select_autoescape

instructions = {
    'base': """{% block instruction %}{% endblock %}""",
    'chat': """{% extends "base" %}{% block instruction %}You are a helpful assistant powered by an LLM.  The input will be the last state of a conversation you are having with a user,  Your response should be directed towards the user.{% endblock %}""",
    'memory': """{% extends "base" %}{% block instruction %}{% endblock %}""",
    'function-calling': """{% extends "base" %}{% block instruction %}{% endblock %}""",
    'summarize': """{% extends "base" %}{% block instruction %}Summarize the text in input{% endblock %}""",
    'describe': """{% extends "base" %}{% block instruction %}Describe in depth the content provided in Input, providing relevant context wherever it may be helpful{% endblock %}""",
    'rephrase': """{% extends "base" %}{% block instruction %}Rephrase the main content that is the topic of the Input, applying any requests/suggestions about how to rephrase{% endblock %}""",
    'persuade': """{% extends "base" %}{% block instruction %}{% endblock %}""",
    'joke': """{% extends "base" %}{% block instruction %}Tell a joke inspired by the topics or themes expressed in the Input{% endblock %}""",
    'plan': """{% extends "base" %}{% block instruction %}Given the task described in the Input, detail a plan of steps to accomplish the task{% endblock %}""",
    'impersonate': """{% extends "base" %}{% block instruction %}Immerse yourself in the role described in the Input{% endblock %}""",
    'poem': """{% extends "base" %}{% block instruction %}You are an expert in writing artistic literature.  Write a poem about the topic or material described in Input{% endblock %}""",
    'user-story': """{% extends "base" %}{% block instruction %}Write a user story about the problem or process described in the Input{% endblock %}""",
    'documentation': """{% extends "base" %}{% block instruction %}Write documentation for the feature or element detailed in the Input{% endblock %}""",
    'explain': """{% extends "base" %}{% block instruction %}Explain in depth the topic provided in Input{% endblock %}""",
    'tone': """{% extends "base" %}{% block instruction %}Analyze the tonality of the text provided in Input{% endblock %}""",
    'intent': """{% extends "base" %}{% block instruction %}Ponder the intent of the text provided in Input{% endblock %}""",
    'assess': """{% extends "base" %}{% block instruction %}Critically assess the text provided in Input and give objective feedback{% endblock %}""",
    'judge': """{% extends "base" %}{% block instruction %}{% block instruction %}{% block instruction %}{% endblock %}""",
    'debate': """{% extends "base" %}{% block instruction %}Debate the viewpoint, belief or statement expressed in Input{% endblock %}""",
    'self-ask': """{% extends "base" %}{% block instruction %}{% endblock %}""",
    'expert-ask': """{% extends "base" %}{% block instruction %}You are preparing to ask an expert for help regarding the topic in Input.  Politely describe to them the issue and what you hope to achieve, and any other details you feel may be relevant{% endblock %}""",
    'transmutate': """{% extends "base" %}{% block instruction %}Create a mockup of Input-- something similar to it in both syntax and nature, but perhaps with a different topic or subjects as to be different{% endblock %}""",
}

env = Environment(
    loader = DictLoader(instructions),
    autoescape = select_autoescape()
)

def getEnv():
    return env

def getInstruction(name):
    return env.get_template(name)

def getRenderedInstruction(name):
    return getInstruction(name).render()