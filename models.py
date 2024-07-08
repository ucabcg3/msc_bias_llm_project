from langchain_community.chat_models import ChatOllama

llama_3_8b = ChatOllama(
    model="llama3:8b", 
    temperature=0,
)

llama_3_70b = ChatOllama(
    model="llama3:70b", 
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

def get_model(model='llama_3_8b'):
    return models[model]

def get_all_models():
    return models