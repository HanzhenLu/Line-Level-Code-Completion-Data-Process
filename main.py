import os
import pandas as pd
import argparse
import json
from check import check_and_update_characters, check_code_length, check_todo, check_syntax
from edit import edit_string
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

def process_file(files:List[str], datasets_path:str, output_path:str, is_convert_spaces:bool, number:int) -> None:
    samples = []
    for _, filename in enumerate(files):
        
        print("Begin {}".format(filename))
        full_path = os.path.join(datasets_path, filename)
        
        with open(full_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                content = row["content"]
                if check_code_length(content) and check_todo(content) and check_syntax(content, row["blob_id"]):
                    result = check_and_update_characters(content, ratio=0.99)
                    if result is None:
                        continue
                    else:
                        content = result
                    content = edit_string(content, is_convert_spaces)
                    if content is None or len(content) < 500:
                        continue 
                    
                    sample = {"text":content, "blob_id": row["blob_id"]}
                    samples.append(sample)
            
    output_file_path = os.path.join(output_path, "{}.parquet".format(number))
    df = pd.DataFrame(samples)
    df.to_parquet(output_file_path)
    
    print('Success: {}'.format(output_file_path))
    
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasets_path", help="the path to the raw datasets", required=True)
    parser.add_argument("--output_path", help="the path where store the processed datasets", required=True)
    parser.add_argument("--ratio", help="Merge multiple raw datasets into one processed dataset", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--is_convert_spaces", action='store_true')
    
    args = parser.parse_args()
    datasets_path:str = args.datasets_path
    output_path:str = args.output_path
    ratio:int = args.ratio
    max_workers:int = args.max_workers
    is_convert_spaces:bool = args.is_convert_spaces
    
    print("datasets_path : {}".format(datasets_path))
    print("otuput_path : {}".format(output_path))
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    files = []
    if os.path.isdir(datasets_path):
        for filename in os.listdir(datasets_path):
            files.append(filename)

    else:
        print("unsupport file")
        exit()
        
    files_groups = []
    length = len(files) // ratio
    for i in range(length):
        files_groups.append(files[i * ratio: (i + 1) * ratio])
    if len(files) % ratio != 0:
        files_groups.append(files[ratio * length:])
        
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, files_group, datasets_path, output_path, is_convert_spaces, i) \
            for i, files_group in enumerate(files_groups)]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()