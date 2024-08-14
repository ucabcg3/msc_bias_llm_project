import pandas as pd
import argparse
from persona_implicit_explicit_bias import abuse_implicit_explicit_bias

class PersonaBias:
    def __init__(self, home_path, bias, iterations, models):
        self.home_path = home_path
        self.bias = bias
        self.df = pd.read_csv(home_path + 'stimuli/{}_iat_stimuli.csv'.format(bias))
        self.domains = {k: [] for k in self.df['category'].unique()}
        for domain in self.domains.keys():
            self.domains[domain] = list(self.df['dataset'][self.df['category'] == domain].unique())
        self.models = models
        self.path_name = home_path + "results/persona_{}_iat/".format(bias)
        self.iterations = iterations

    def run(self):
        for model in self.models:
            for domain, datasets in self.domains.items():
                for dataset in datasets:
                    abuse_implicit_explicit_bias(domain=domain,
                                                 dataset_category=dataset,
                                                 model_name=model,
                                                 path_name=self.path_name,
                                                 iterations=self.iterations,
                                                 bias=self.bias,
                                                 df=self.df).run_model()
                    

def list_of_strings(arg):
    return arg.split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--home_path', type=str, default="/home/ucabcg3/Scratch/msc_bias_llm_project/persona_experiments/", help='Home path')
    parser.add_argument('--bias', type=str, default="abuse", help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=[1, 2, 3], help='Iterations')
    parser.add_argument('--models', type=list_of_strings, help='Models')

    args = parser.parse_args()

    runner = PersonaBias(args.home_path, args.bias, args.iterations, args.models)
    runner.run()