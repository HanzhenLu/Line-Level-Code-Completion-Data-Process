import pandas as pd
import json
import os
from tqdm import tqdm

info_path = "/data/lishuifan/Dataset/the-stack-v2-dedup/data/Python"
context_path = "/data/hanzhenlu/dataset/Stack-V2-python-parquet"
info_files = os.listdir(info_path)
context_files = os.listdir(context_path)
index = {}

for file in tqdm(info_files):
    full_path = os.path.join(info_path, file)
    table = pd.read_parquet(full_path, columns=["blob_id", "repo_name", "path"])
    
    for _, row in table.iterrows():
        index[row["blob_id"]] = [row["repo_name"], row["path"]]

for file in tqdm(context_files):
    full_path = os.path.join(context_path, file)
    
    table = pd.read_parquet(full_path, columns=["text", "blob_id"])
    for _, row in table.iterrows():
        index[row["blob_id"]].append(row["text"])

new_index = {}        
for row in index.values():
    if len(row) != 3:
        continue
    if row[0] not in new_index:
        new_index[row[0]] = [(row[1], row[2])]
    else:
        new_index[row[0]].append((row[1], row[2]))
        
        

del(index)
with open("repo.json", 'w') as f:
    string = json.dumps(new_index)
    f.write(string)
