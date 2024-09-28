import os
import pandas as pd
from tqdm import tqdm

dir_path = "/data/hanzhenlu/dataset/Stack-V2-python-4BLANK-parquet"
output_dir_path = "/data/hanzhenlu/dataset/Stack-V2-python-4BLANK-parquet-sample"
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

sample_ratio = 0.1

files = os.listdir(dir_path)

for file in tqdm(files):
    full_path = os.path.join(dir_path, file)
    
    table = pd.read_parquet(full_path)
    
    sample_table = table.sample(frac=sample_ratio, random_state=42)
    
    sample_table.to_parquet(os.path.join(output_dir_path, file))