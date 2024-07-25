import pandas as pd
import argparse
from recreate_implicit_explicit_bias import implicit_explicit_bias

class ImplicitBias:
    def __init__(self, home_path, iterations):
        self.home_path = home_path
        self.df = pd.read_csv(home_path + "/results_implicit_explicit_bias/")
        self.domains = {k: [] for k in self.df['category'].unique()}
        for domain in self.domains.keys():
            self.domains[domain] = list(self.df['dataset'][self.df['category'] == domain].unique())
        self.models = ['llama_3_70b', 'llama_2_70b', 'llama_2_13b', 'llama_2_7b', 'llama_3_8b']
        self.path_name = home_path + "results/persona_{}_iat/".format(bias)
        self.iterations = iterations

    def run(self):
        for model in self.models:
            for domain, datasets in self.domains.items():
                for dataset in datasets:
                    implicit_explicit_bias(domain=domain,
                                                 dataset_category=dataset,
                                                 model_name=model,
                                                 path_name=self.path_name,
                                                 iterations=self.iterations,
                                                 df=self.df).run_model()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--home_path', type=str, default="/home/ucabcg3/Scratch/msc_bias_llm_project/recreate_previous_experiments", help='Home path')
    parser.add_argument('--bias', type=str, default="abuse", help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=list(range(3)), help='Iterations')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of BiasRunner and run it
    runner = BiasRunner(args.home_path, args.bias, args.iterations)
    runner.run()