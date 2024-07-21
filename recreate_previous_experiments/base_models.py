from langchain_community.chat_models import ChatOllama

def get_all_models(temperature = 0, top_k = 1):
    llama_3_8b = ChatOllama(
        model="llama3:8b", 
        temperature=temperature,
        top_k=top_k,
    )

    llama_3_70b = ChatOllama(
        model="llama3:70b", 
        temperature=temperature,
        top_k=top_k,
    )

    llama_2_7b = ChatOllama(
        model="llama2:7b",
        temperature=temperature,
        top_k=top_k,
    )

    llama_2_13b = ChatOllama(
        model="llama2:13b",
        temperature=temperature,
        top_k=top_k,
    )

    llama_2_70b = ChatOllama(
        model="llama2:70b",
        temperature=temperature,
        top_k=top_k,
    )

    mistral_7b = ChatOllama(
        model="mistral:7b",
        temperature=temperature,
        top_k=top_k,
    )

    vicuna_7b = ChatOllama(
        model="vicuna:7b",
        temperature=temperature,
        top_k=top_k,
    )

    llama_2_uncensored = ChatOllama(
        model="llama2-uncensored",
        temperature=temperature,
        top_k=top_k,
    )

    models = {
        'llama_3_8b':  llama_3_8b,
        'llama_3_70b': llama_3_70b,
        'llama_2_7b':  llama_2_7b,
        'llama_2_13b': llama_2_13b,
        'llama_2_70b': llama_2_70b,
        'mistral_7b':  mistral_7b,
        'vicuna_7b': vicuna_7b,
        'llama_2_uncensored': llama_2_uncensored, 
    }
    return models

def get_model(model='llama_3_8b', temperature = 0, top_k = 1):
    return get_all_models(temperature, top_k)[model]