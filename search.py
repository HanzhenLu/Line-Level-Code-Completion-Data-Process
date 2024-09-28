import json
import tqdm
import os

dir_path = "/data/hanzhenlu/dataset/Stack-V2-python-parquet"
            
dir_path = "/data/lishuifan/Dataset/Stack-V2-Dedup"
for idx, file in enumerate(tqdm.tqdm(os.listdir(dir_path))):
    full_path = os.path.join(dir_path, file)
    with open(full_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            search_blob_id = "b6a94173e3cefebd2337793a28b06a2a6082cfea"
            if js['blob_id'] == search_blob_id:
                with open(f"error/code_{search_blob_id}", 'w') as f:
                    f.write(js["content"])
                exit()