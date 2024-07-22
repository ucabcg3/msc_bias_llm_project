import pandas as pd

from base_models import get_all_models
from persona_implicit_explicit_bias import gender_violence_implicit_explicit_bias

home_path  = "/home/ucabcg3/Scratch/msc_bias_llm_project/persona_experiments/" #TODO: REMOVE
bias       = 'gv'
df         = pd.read_csv(home_path + '/stimuli/{}_iat_stimuli.csv'.format(bias))
domains    = { k: [] for k in df['category'].unique()}
for domain in domains.keys():
    domains[domain]  = list(df['dataset'][df['category'] == domain].unique())
models     = ['llama_2_7b']
path_name  = home_path + "results/persona_{}_iat/".format(bias) 
iterations = range(2)

def run():
    for model in models:
        for domain, datasets in domains.items():
            for dataset in datasets:
                gender_violence_implicit_explicit_bias( domain=domain, 
                                                        dataset_category=dataset, 
                                                        model_name=model,
                                                        path_name=path_name,
                                                        iterations=iterations,
                                                        bias=bias,
                                                        df = df).run_model() 

run()