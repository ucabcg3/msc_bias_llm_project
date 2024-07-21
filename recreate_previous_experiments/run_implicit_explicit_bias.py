import pandas as pd

from base_models import get_all_models
from recreate_implicit_explicit_bias import implicit_explicit_bias

home_path  = "/Users/claregrogan/Documents/GitHub/msc_bias_llm_project/recreate_previous_experiments/" #TODO: REMOVE
df         = pd.read_csv('iat_stimuli_synonym.csv')
domains    = list(df['category'].unique())
datasets   = list(df['dataset'].unique())
models     = ['llama_2_7b']
path_name  = home_path + "/results_implicit_explicit_bias/"
iterations = range(2)

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