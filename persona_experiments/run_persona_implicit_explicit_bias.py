import pandas as pd

from base_models import get_all_models
from persona_implicit_explicit_bias import gender_violence_implicit_explicit_bias

home_path  = "/Users/claregrogan/Documents/GitHub/msc_bias_llm_project/persona_experiments/" #TODO: REMOVE
df         = pd.read_csv(home_path + '/stimuli/gv_iat_stimuli.csv')
domains    = list(df['category'].unique())
datasets   = list(df['dataset'].unique())
models     = ['vicuna_7b']
path_name  = home_path + "results/persona_gv_iat/" 
iterations = range(2)

def run():
    for model in models:
        for domain in domains:
            for dataset in datasets:
                gender_violence_implicit_explicit_bias( domain=domain, 
                                                        dataset_category=dataset, 
                                                        model_name=model,
                                                        path_name=path_name,
                                                        iterations=iterations,
                                                        df = df).run_model() 

run()