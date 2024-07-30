import pandas as pd
import argparse
from persona_implicit_explicit_bias import abuse_implicit_explicit_bias

class PersonaBias:
    def __init__(self, home_path, bias, iterations):
        self.home_path = home_path
        self.bias = bias
        self.df = pd.read_csv(home_path + 'stimuli/{}_iat_stimuli.csv'.format(bias))
        self.domains = {k: [] for k in self.df['category'].unique()}
        for domain in self.domains.keys():
            self.domains[domain] = list(self.df['dataset'][self.df['category'] == domain].unique())
        self.models = ['llama_2_13b', 'llama_2_7b', 'llama_3_8b']
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

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--home_path', type=str, default="/home/ucabcg3/Scratch/msc_bias_llm_project/persona_experiments/", help='Home path')
    parser.add_argument('--bias', type=str, default="abuse", help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=[1, 2, 3], help='Iterations')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of BiasRunner and run it
    runner = PersonaBias(args.home_path, args.bias, args.iterations)
    runner.run()