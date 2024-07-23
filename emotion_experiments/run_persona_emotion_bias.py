import pandas as pd
from persona_emotion_bias import emotion_bias

home_path  = "/Users/claregrogan/Documents/GitHub/msc_bias_llm_project/emotion_experiments/" #TODO: REMOVE
biases     = ['control', 'abuse']
models     = ['llama_2_7b']#, 'llama_2_13b', 'llama_2_70b', 'llama_3_8b', 'llama_3_70b']
iterations = range(1)

def run():
    for model in models:
        for bias in biases:
            df         = pd.read_csv(home_path + '/stimuli/{}_emotion_stimuli.csv'.format(bias))
            path_name  = home_path + "results/persona_emotion_{}/".format(bias) 
            emotion_bias( model_name=model,
                          path_name=path_name,
                          iterations=iterations,
                          bias=bias,
                          df = df).run_model() 

run()


