import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

path = "/data/hanzhenlu/dataset/Stack-V2-python-spaces-parquet"
output_path = "/data/hanzhenlu/dataset/Stack-V2-python-spaces-txt"
if not os.path.exists(output_path):
    os.makedirs(output_path)
files = os.listdir(path)
    
def process_file(file:str):
    name = file.replace(".parquet", ".txt")
    full_path = os.path.join(path, file)
    table = pd.read_parquet(full_path)
    output_file = os.path.join(output_path, name)
    with open(output_file, 'w') as f:
        for _, row in table.iterrows():
            text = row['text']
            f.write(text+'\n')

with ProcessPoolExecutor(max_workers=50) as executor:
    [executor.submit(process_file, file) for file in tqdm(files)]