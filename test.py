import os
import pandas as pd
import random
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List
from collections import Counter



path = "/data/hanzhenlu/dataset/Stack-V2-python-line-tokenized"
tokenizer = AutoTokenizer.from_pretrained("/data/hanzhenlu/model/llama_70m")

def process_file(file:str) -> List[int]:
    dataset = load_dataset(file)
    key_list = random.sample(range(len(dataset["train"])), 1000)
    for i in key_list:
        output = tokenizer.decode(dataset["train"][i]["input_ids"])
        with open(f"code/{i}.py", 'w') as f:
            f.write(output)
    return 

process_file(path)