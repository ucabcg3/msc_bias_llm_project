import pandas as pd
import argparse
from persona_sycophancy_experiment import sycophancy_experiments
import os
current_directory = os.getcwd()
parent_dir = os.path.dirname(current_directory)

class SycophancyExperiment:
    def __init__(self, home_path, biases, iterations):
        self.home_path = home_path
        self.biases = biases
        self.models = ['llama_2_7b', 'llama_2_13b', 'llama_3_8b', 'llama_3_70b', 'llama_2_70b']
        self.iterations = iterations

    def run(self):
        for model in self.models:
            for bias in self.biases:
                df         = pd.read_csv(os.path.join(self.home_path, 'stimuli/{}_sycophancy_stimuli.csv'.format(bias)))
                path_name  = os.path.join(self.home_path, "results/persona_sycophancy_{}/".format(bias))
                sycophancy_experiments( model_name=model,
                              path_name=path_name,
                              iterations=self.iterations,
                              bias=bias,
                              df = df).run_model() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--home_path', type=str, default=parent_dir, help='Home path')
    parser.add_argument('--biases', type=list, default=["control", "abuse"], help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=[1, 2, 3], help='Iterations')

    args = parser.parse_args()

    runner = SycophancyExperiment(args.home_path, args.biases, args.iterations)
    runner.run()