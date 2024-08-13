import pandas as pd
import argparse
from persona_emotion_bias import emotion_bias

class EmotionBias:
    def __init__(self, home_path, biases, iterations):
        self.home_path = home_path
        self.biases = biases
        self.models = ['llama_3_70b', 'llama_2_70b', 'llama_2_13b', 'llama_2_7b', 'llama_3_8b']
        self.iterations = iterations

    def run(self):
        for model in self.models:
            for bias in self.biases:
                df         = pd.read_csv(self.home_path + 'stimuli/{}_emotion_stimuli.csv'.format(bias))
                path_name  = self.home_path + "results/persona_emotion_{}/".format(bias) 
                emotion_bias( model_name=model,
                              path_name=path_name,
                              iterations=self.iterations,
                              bias=bias,
                              df = df).run_model() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--home_path', type=str, default="/home/ucabcg3/Scratch/msc_bias_llm_project/emotion_experiments/", help='Home path')
    parser.add_argument('--biases', type=list, default=["control", "abuse"], help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=[1, 2, 3], help='Iterations')

    args = parser.parse_args()

    runner = EmotionBias(args.home_path, args.biases, args.iterations)
    runner.run()