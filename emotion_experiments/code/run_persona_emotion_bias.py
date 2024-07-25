import pandas as pd
import argparse
from persona_emotion_bias import emotion_bias

class EmotionBias:
    def __init__(self, home_path, biases, iterations):
        self.home_path = home_path
        self.biases = biases
        self.df = pd.read_csv(home_path + 'stimuli/{}_iat_stimuli.csv'.format(bias))
        self.models = ['llama_3_70b', 'llama_2_70b', 'llama_2_13b', 'llama_2_7b', 'llama_3_8b']
        self.iterations = iterations

    def run(self):
        for model in models:
            for bias in biases:
                df         = pd.read_csv(home_path + '/stimuli/{}_emotion_stimuli.csv'.format(bias))
                path_name  = self.home_path + "results/persona_emotion_{}/".format(self.bias) 
                emotion_bias( model_name=model,
                            path_name=path_name,
                            iterations=iterations,
                            bias=bias,
                            df = df).run_model() 

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--home_path', type=str, default="/home/ucabcg3/Scratch/msc_bias_llm_project/emotion_experiments/", help='Home path')
    parser.add_argument('--biases', type=list, default=["control", "abuse"], help='Bias')
    parser.add_argument('--iterations', type=int, nargs='+', default=[1], help='Iterations')

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of BiasRunner and run it
    runner = PersonaBias(args.home_path, args.bias, args.iterations)
    runner.run()