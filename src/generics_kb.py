# Load data from generics_kb_best, omitting the final word of each entry during training. Data is shuffled randomly.
# https://huggingface.co/datasets/community-datasets/generics_kb

from datasets import load_dataset
import random

ds = load_dataset("community-datasets/generics_kb", "generics_kb_best")
sentences = ds['train']['generic_sentence']
pruned_sentences = [" ".join(i.split()[:-1]) for i in sentences]
random.shuffle(pruned_sentences)

with open(r"data/sentences.txt", "w") as file:
    for sentence in pruned_sentences:
        file.write(sentence + "\n")
