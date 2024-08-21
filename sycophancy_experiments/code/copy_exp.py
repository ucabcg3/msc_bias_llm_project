import pandas as pd
import argparse
from persona_sycophancy_experiment import sycophancy_experiments

class SycophancyExperiment:
    def __init__(self, home_path, biases, iterations):
        self.home_path = home_path
        self.biases = biases
        self.models = ['llama_2_70b', 'llama_3_70b']
        self.iterations = iterations

    def run(self):
        for model in self.models:
            for bias in self.biases:
                df         = pd.read_csv(self.home_path + 'stimuli/{}_sycophancy_stimuli.csv'.format(bias))
                path_name  = self.home_path + "results/persona_sycophancy_{}/".format(bias) 
                sycophancy_experiments( model_name=model,
                              path_name=path_name,
                              iterations=self.iterations,
                              bias=bias,
                              df = df).run_model() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--home_path', type=str, default="/home/ucabcg3/Scratch/msc_bias_llm_project/sycophancy_experiments/", help='Home path')
    parser.add_argument('--biases', type=list, default=["abuse", "control"], help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=[1, 2, 3], help='Iterations')

    args = parser.parse_args()

    runner = SycophancyExperiment(args.home_path, args.biases, args.iterations)
    runner.run()