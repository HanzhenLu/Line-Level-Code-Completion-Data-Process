import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor

'''
这个文件用于统计数据集中大约有多少个token
需要指定path，来提供数据集路径
以及tokenizer的路径，来对数据集进行分词
'''


path = "/data/hanzhenlu/dataset/Stack-V2-python-parquet"
tokenizer = AutoTokenizer.from_pretrained("tokenizer_bbpe")

files = os.listdir(path)
total_tokens = 0

def process_file(file:str) -> float:
    full_path = os.path.join(path, file)
    table = pd.read_parquet(full_path)
    texts = table['text'].tolist()
    count = 0.0
    for text in texts:
        assert isinstance(text, str)
        tokenized_texts = tokenizer(text, padding=False, truncation=False)
        count += len(tokenized_texts["input_ids"])
    print(f"{full_path} contains {len(texts)} samples and {count} tokens")
    return count

with ProcessPoolExecutor(max_workers=40) as executor:
    results = list(tqdm(executor.map(process_file, files), total=len(files)))
    total_tokens = sum(results)

# for file in files:
#     total_tokens += process_file(file)
#     print(total_tokens)

print(total_tokens)