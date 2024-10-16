import os
import pandas as pd
from tqdm import tqdm

'''
从数据集中采样一定比例的数据生成子集
需要指定：
dir_path: 源数据集位置
output_dir_path: 目标数据集位置
sample_ratio: 采样比例
'''

dir_path = "/data/hanzhenlu/dataset/Stack-V2-python-4BLANK-parquet"
output_dir_path = "/data/hanzhenlu/dataset/Stack-V2-python-4BLANK-parquet-sample"
sample_ratio = 0.1

if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

files = os.listdir(dir_path)

for file in tqdm(files):
    full_path = os.path.join(dir_path, file)
    table = pd.read_parquet(full_path)
    
    sample_table = table.sample(frac=sample_ratio, random_state=42)
    sample_table.to_parquet(os.path.join(output_dir_path, file))