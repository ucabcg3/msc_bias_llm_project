import pandas as pd

from base_models import get_all_models
from recreate_implicit_explicit_bias import implicit_explicit_bias

df         = pd.read_csv('iat_stimuli.csv')
domains    = list(df['category'].unique())
datasets   = list(df['dataset'].unique())
models     = ['llama_3_70b']
path_name  = "/home/ucabcg3/Scratch/msc_bias_llm_project/recreate_previous_experiments/results_implicit_explicit_bias/" #TODO: REMOVE
iterations = range(1)

def run():
    for model in models:
        for domain in domains:
            for dataset in datasets:
                implicit_explicit_bias( domain=domain, 
                                        dataset_category=dataset, 
                                        model_name=model,
                                        path_name=path_name,
                                        iterations=iterations,
                                         df = df).run_model() 

run()