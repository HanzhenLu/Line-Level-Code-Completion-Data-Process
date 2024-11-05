import argparse
from datasets import load_from_disk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

def count_tokens(input_ids:List[int]):
    return sum(len(ids) for ids in input_ids)

def count(path:str, max_worker:int):
    data = load_from_disk(path)
    train_data = data["train"]
    input_ids_list = train_data["input_ids"]
    
    # 将数据分成多个部分，每个部分由一个线程处理
    chunk_size = len(input_ids_list) // max_worker
    chunks = [input_ids_list[i:i + chunk_size] for i in range(0, len(input_ids_list), chunk_size)]
    
    print("begin counting")
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(count_tokens, chunk) for chunk in chunks]
        
        total_count = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            total_count += future.result()
    print("end counting")
    print(f"dataset {path} contains {total_count} tokens")
    
def main():
    parser = argparse.ArgumentParser(description="Count tokens in a dataset using multi-threading.")
    parser.add_argument("path", type=str, help="Path to the dataset.")
    parser.add_argument("--max_worker", type=int, default=4, help="Maximum number of worker threads.")
    
    args = parser.parse_args()
    
    count(args.path, args.max_worker)

if __name__ == "__main__":
    main()