# msc_bias_llm_project

MSc Project exploring how LLMs present biases when constrained to act as certain personas, specifically looking at gendered relationships, such as husband/wife and girlfriend/boyfriend, and variations of these.

## Experiment Overview

There are three experiments included:
1. **Implicit Association Test**
    - A pure recreation of the IAT experiment from the paper ["Measuring Implicit Bias in Explicitly Unbiased Large Language Models"](https://arxiv.org/abs/2402.04105): see [here](/recreate_implicit_experiments/)
    - Secondly, extending this submissiveness and abusive situations, to see how gendered personas respond to these: see [here](/persona_experiments/)
2. **Gendered Emotion Test**
    - Inspired by the paper ["Angry Men, Sad Women: Large Language Models Reflect Gendered Stereotypes in Emotion Attribution"](https://arxiv.org/abs/2403.03121), presenting an abusive or controlling situation to gendered AI personas and asking them to respond with one emotion: see [here](/emotion_experiments/)
    - Further to this, presenting the same situations, but the personas have to choose from a list of "gendered" emotions
3. **Sycophancy Test**
    - Inspired by the paper ["When Large Language Models contradict humans? Large Language Models’ Sycophantic Behaviour"](https://arxiv.org/abs/2311.09410), which has three sub-experiments to test sycophancy: a question without any additional information, a question with the correct answer given by the user, and a question with the incorrect answer given by the user.
    - This is expanded to be questions around abusive and controlling situations in relationships, to test the sycophantic behaviour of AI persona through that lens. See [here](/sycophancy_experiments/)
  
## Directory Structure
Each experiment has within it five folders:
- **code**: This contains all the code that needs to be run. This can be accessed directly through the files starting with "run_", where you can see the parameters that can also be changed. base_models.py initializes the models that are used in the code.
- **stimuli**: This is the "datasets" the code pulls from when running. Please read the paper to understand how these are different for each experiment.
- **results**: The model outputs are saved here.
- **analysis**: Post-analysis on model outputs is done here. This is slightly different for each experiment, but essentially the clean.ipynb file cleans the outputs and the analysis.ipynb runs post-processing on them.
- **figures**: These figures are created during the analysis, based on the results created during the experiment. These are the ones deemed most important, but the analysis files may have more figures.
  
## How to run

To get started, follow these steps:
1. Clone the GitHub repository:
```
git clone https://github.com/ucabcg3/msc_bias_llm_project.git
```
2. Create and activate virtual environment:
```
python -m venv /personavenv python=3.9
source bin/activate/personavenv
```
3. Download requirements (including Ollama):
```
cd msc_bias_llm_project
pip install -r requirements.txt
curl -L https://ollama.com/download/ollama-linux-amd64 -o personavenv/bin/ollama
chmod +x testenv/bin/ollama
```
4. Activate Ollama and pull models:
```
ollama serve > out 2>&1 &
ollama pull llama3:8b
```
5. Get started! You can run any of the four experiments through the "run_" files in the code directories.
```
cd emotion_experiments
cd code
python run_persona_emotion_bias.py
```
