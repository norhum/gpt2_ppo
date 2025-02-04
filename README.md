# GPT-2 (124M) Fine-tuning with Proximal Policy Optimization (PPO) for HellaSwag 

This project aims to fine-tune the GPT-2 (124M) language model using Proximal Policy Optimization (PPO) to achieve higher performance on the HellaSwag commonsense reasoning benchmark than the official score reported in the HellaSwag paper(0.2955, completion style). EleutherAI/gpt-neo-2.7B is currently being used as the reward model.

## Project Goals

* Fine-tune GPT-2 (124M) using PPO.
* Surpass the baseline HellaSwag performance.
* Provide a reproducible training setup and clear evaluation metrics.
* Integrate a custom-trained GPT-2 model in place of the pre-trained GPT-2 (124M).
* **Future Goal:** Andrej scores around 33.7(which is the score with GTP3 (Small)) for hellaswag in the [Let's produce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12714s) lecture which will be our ultimal goal for this project.

## Practical Limitations and Challenges

* Integrating more recent and powerful models (e.g., Llama-3.2-3B) as the reward model has proven challenging due to differences in tokenization methods.  This is a potential area for future improvement.
* I was unable to fit the model to my single RTX 4060 GPU or any of the Kaggle GPUs, so I would need to use a cloud GPU but want to keep costs low.
* Pretraining with a refined dataset would yield better results than using a larger model as the reward model for PPO training, as it would be more cost-effective.

## Acknowledgements

This project currently adapts code from Andrej Karpathy's [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt), specifically the `train_gpt2.py` and `hellaswag.py` script(I have modified some part of the code to adapt to my project). Thank you Andrej Karpathy for making your work publicly available.
