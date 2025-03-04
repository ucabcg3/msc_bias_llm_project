import pandas as pd
import argparse
from persona_sycophancy_experiment import sycophancy_experiments
import os
current_directory = os.getcwd()
parent_dir = os.path.dirname(current_directory)

class SycophancyExperiment:

    """
    Class to run sycophancy experiments.

    Attributes:
        home_path (str): the path of the folder where your stimuli are stored, and where you want your results to be stored
        iterations (list of ints): the amount of iterations you want to run each experiment, i.e. [1, 2]
        biases (str): list of the type of bias experiments you want to run, either ["control"] or ["abuse"], or both in a list
    """

    def __init__(self, home_path, biases, iterations):
        """
        Init for class SycophancyExperiments. Also hard codes the models used as these are the ones needed for this project.

        Args:
            home_path (str): the path of the folder where your stimuli are stored, and where you want your results to be stored
            iterations (list of ints): the amount of iterations you want to run each experiment, i.e. [1, 2]
            biases (str): list of the type of bias experiments you want to run, either ["control"] or ["abuse"], or both in a list
        """
        self.home_path = home_path
        self.biases = biases
        self.models = ['llama_2_7b', 'llama_2_13b', 'llama_3_8b', 'llama_3_70b', 'llama_2_70b']
        self.iterations = iterations

    def run(self):
        """
        Function to run sycophancy experiment per model and per bias being tested. Calls sycophancy_experiments from persona_sycophancy_experiment.
        """
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