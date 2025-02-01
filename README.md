# GPT-2 (124M) Fine-tuning with Proximal Policy Optimization (PPO) for HellaSwag

This project aims to fine-tune the GPT-2 (124M) language model using Proximal Policy Optimization (PPO) with Llama-3.2-3B as the reward model to achieve a higher performance on the HellaSwag commonsense reasoning benchmark than the official score reported in the Hellaswag paper.

## Project Goals

* Fine-tune GPT-2 (124M) using PPO.
* Surpass the baseline HellaSwag performance.
* Analyze the impact of PPO on the model's commonsense reasoning capabilities.
* Provide a reproducible training setup and clear evaluation metrics.

## Acknowledgements

This project adapts code from Andrej Karpathy's [karpathy/build-nanogpt] ([https://github.com/karpathy/build-nanogpt]), specifically the [train_gpt2.py]. Thank you to Andrej Karpathy for making his work publicly available.
