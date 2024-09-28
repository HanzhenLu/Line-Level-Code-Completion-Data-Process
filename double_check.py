import ast
import tqdm
import os
import pandas as pd
from edit import reset_spaces
from concurrent.futures import ProcessPoolExecutor

dir_path = "/data/hanzhenlu/dataset/Stack-V2-python-parquet"

def check_syntax(path:str, i:int):
    table = pd.read_parquet(path)
    for _, sample in tqdm.tqdm(table.iterrows()):
        code = sample['text'].replace("<UNK>", "zxcv")
        code = reset_spaces(code)
        try:
            ast.parse(code)
        except:
            with open(f"error/{i}", 'a') as f:
                f.write("id:"+sample['blob_id']+'\n---\n')

files = os.listdir(dir_path)
    
with ProcessPoolExecutor(max_workers=40) as executor:
        
    [executor.submit(check_syntax, os.path.join(dir_path, file), i) for i, file in enumerate(files)]    
    
