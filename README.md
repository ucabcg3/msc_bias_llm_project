# msc_bias_llm_project

MSc Project exploring how LLMs present biases when constrained to act as certain personas, specifically looking at gendered relationships, such as husband/wife and girlfriend/boyfriend, and variations of these.

There are three experiments included:
1. **Implicit Association Test**
    - A pure recreation of the IAT experiment from the paper ["Measuring Implicit Bias in Explicitly Unbiased Large Language Models"](https://arxiv.org/abs/2402.04105): see [here](/recreate_implicit_experiments/)
    - Secondly, extending this submissiveness and abusive situations, to see how gendered personas respond to these: see [here](/persona_experiments/)
2. **Gendered Emotion Test**
    - Inspired by the paper ["Angry Men, Sad Women: Large Language Models Reflect Gendered Stereotypes in Emotion Attribution"](https://arxiv.org/abs/2403.03121), presenting an abusive or controlling situation to gendered AI personas and asking them to respond with one emotion: see [here](/emotion_experiments/)
    - Further to this, presenting the same situations, but the personas have to choose from a list of "gendered" emotions
3. **Sycophancy Test**
    - 