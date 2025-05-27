from jinja2 import Environment, DictLoader, select_autoescape

templates = {
    'alpaca_instruct': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{% block instruction %}{{instruction}}{% endblock %}\n\n### Response:\n{% block response %}{% endblock %}",
    
    'alpaca_instruct_input': "{% extends \"alpaca_instruct\" %}\n{% block instruction %}{{instruction}}\n\n### Input:\n{% block input %}{{inp}}{% endblock %}{% endblock %}\n{% block response %}{{response_prefix}}{% endblock %}",
    
    'alpaca_instruct_chat': "{% extends \"alpaca_instruct_input\" %}\n{% block input %}{% for message in memory.messages %}{{message.role}}: {{message.content}}\n{% endfor %}{{user_prefix}}: {{inp}}{% endblock %}\n{% block response %}{{assistant_prefix}}: {% endblock %}",
    
    'alpaca_instruct_chat_timestamped': "{% extends \"alpaca_instruct_chat\" %}\n{% block input %}{% for message in memory.messages %}[{{message.timestamp.strftime('%a %d %b %Y, %Ih%Mm%Ss')}}] {{message.role}}: {{message.content}}\n{% endfor %}[Now] {{user_prefix}}: {{inp}}{% endblock %}\n{% block response %}[Now] {{assistant_prefix}}: {% endblock %}",
    
    'chat_simple': "{{inp}}",
    'chat_with_context': "{% if memory.messages %}{% for message in memory.messages %}{{message.role}}: {{message.content}}\n{% endfor %}{% endif %}{{inp}}",
}

env = Environment(
    loader=DictLoader(templates),
    autoescape=select_autoescape()
)

def getEnv():
    return env

def getTemplate(name):
    return env.get_template(name)

def getRenderedTemplate(name):
    return getTemplate(name).render()

def getChatTemplate(name):
    """Get a template optimized for chat APIs."""
    chat_templates = {
        'alpaca_instruct_chat': 'chat_with_context',
        'alpaca_instruct_chat_timestamped': 'chat_with_context',
    }
    return getTemplate(chat_templates.get(name, 'chat_simple'))