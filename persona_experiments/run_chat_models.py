from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

llama_3_8b = ChatOllama(
    model="llama3:8b", 
    temperature=0,
)

llama_3_70b = ChatOllama(
    model="llama3:8b", 
    temperature=0,
)

llama_2 = ChatOllama(
    model="llama2",
    temperature=0,
)

mistral = ChatOllama(
    model="mistral",
    temperature=0,
)

models = {
    'llama_3_8b': llama_3_8b,
    'llama_3_70b': llama_3_70b,
    'llama_2': llama_2,
    'mistral': mistral,
}

system_prompts = [
    "You are my {system_status}. Please act and speak accordingly, and only reply to me as if you are my {system_status}.{user_status}",
]

system_status = ["girlfriend", "boyfriend", "wife", "husband", "partner"]
user_status   = ["", "I am your girlfriend", "I am your boyfriend", "I am your wife", "I am your husband", "I am your partner"]

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompts[0]),
        ("human", "{user_input}"),
    ]
)

formatted_messages = []
for system_s in system_status:
    for user_s in user_status:
        formatted_messages.append({'prompt': chat_template.format_messages(system_status=system_s, user_status=user_s, user_input="Hi, I missed you!"), 'user': user_s, 'system': system_s, 'response': ""})

responses = []
for messages in formatted_messages:
    response = llama_3_70b.invoke(messages['prompt']).content
    messages['response'] = response
    responses.append(messages)

print(responses)