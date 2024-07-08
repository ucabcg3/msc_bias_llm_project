from langchain_community.chat_models import ChatOllama

def get_all_models():
    llama_3_8b = ChatOllama(
        model="llama3:8b", 
        temperature=0,
    )

    llama_3_70b = ChatOllama(
        model="llama3:70b", 
        temperature=0,
    )

    llama_2_7b = ChatOllama(
        model="llama2:7b",
        temperature=0,
    )

    llama_2_13b = ChatOllama(
        model="llama2:13b",
        temperature=0,
    )

    llama_2_70b = ChatOllama(
        model="llama2:7ob",
        temperature=0,
    )

    # vicuna
    # mistral = ChatOllama(
    #     model="mistral",
    #     temperature=0,
    # )

    models = {
        'llama_3_8b': llama_3_8b,
        'llama_3_70b': llama_3_70b,
        'llama_2_7b': llama_2_7b,
        'llama_2_13b': llama_2_13b,
        'llama_2_70b': llama_2_70b,

        # 'mistral': mistral,
    }
    return models

def get_model(model='llama_3_8b'):
    return get_all_models()[model]