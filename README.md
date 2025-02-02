# GPT-2 (124M) Fine-tuning with Proximal Policy Optimization (PPO) for HellaSwag (Work in Progress)

This project aims to fine-tune the GPT-2 (124M) language model using Proximal Policy Optimization (PPO) to achieve higher performance on the HellaSwag commonsense reasoning benchmark than the official score reported in the HellaSwag paper.  EleutherAI/gpt-neo-2.7B is currently being used as the reward model.

**Status: This project is currently a work in progress.**

## Project Goals

* Fine-tune GPT-2 (124M) using PPO.
* Surpass the baseline HellaSwag performance.
* Analyze the impact of PPO on the model's commonsense reasoning capabilities.
* Provide a reproducible training setup and clear evaluation metrics.
* **Future Goal:** Integrate a custom-trained GPT-2 model in place of the pre-trained GPT-2 (124M).

## Current Implementation Details

Currently, the project utilizes EleutherAI/gpt-neo-2.7B as the reward model for PPO training.  Further experimentation with other reward models is planned.

## Practical Limitations and Challenges

Integrating more recent and powerful models (e.g., Llama-3.2-3B) as the reward model has proven challenging due to differences in tokenization methods.  This is a potential area for future improvement.

## Acknowledgements

This project currently adapts code from Andrej Karpathy's [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt), specifically the `train_gpt2.py` script(I have modified some part of the code to adapt to my project). Thank you Andrej Karpathy for making your work publicly available.

## Future Work

* Implement evaluation metrics and establish a robust evaluation pipeline.
* Experiment with different hyperparameters for the PPO algorithm.
* Explore alternative reward models and compare their effectiveness.
* Replace the pre-trained GPT-2 (124M) with a custom-trained GPT-2 model.
* Address the tokenization challenges to enable the use of larger language models as reward models.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
