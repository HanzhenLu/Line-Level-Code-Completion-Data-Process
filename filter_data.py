import os
import datasets
import random
import argparse
from vllm import LLM, SamplingParams
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
def deepseek_process_batch(batch, llm:LLM, sampling_params:SamplingParams, only_prefix:bool):
    valid_batch = {"prefix":[], "suffix":[], "middle":[]}
    for i, m in enumerate(batch["middle"]):
        if m != "":
            valid_batch["prefix"].append(batch["prefix"][i])
            valid_batch["middle"].append(batch["middle"][i])
            valid_batch["suffix"].append(batch["suffix"][i])
    # 将数据转换为deepseek-coder支持的格式
    if not only_prefix:
        context = ["<｜fim▁begin｜>" + p + "<｜fim▁hole｜>" + s + "<｜fim▁end｜>" for p, s in zip(valid_batch["prefix"], valid_batch["suffix"])]
    else:
        context = ["<｜fim▁begin｜>" + p + "<｜fim▁hole｜><｜fim▁end｜>" for p in valid_batch["prefix"]]
    generated_results = llm.generate(context, sampling_params, use_tqdm=False)
    generated_code = [[output.text for output in generated_result.outputs] for generated_result in generated_results]
    valid_batch["prediction"] = generated_code
    return valid_batch

def llama_process_batch(batch, llm:LLM, sampling_params:SamplingParams, only_prefix:bool):
    valid_batch = {"prefix":[], "suffix":[], "middle":[]}
    for i, m in enumerate(batch["middle"]):
        if m != "":
            valid_batch["prefix"].append(batch["prefix"][i])
            valid_batch["middle"].append(batch["middle"][i])
            valid_batch["suffix"].append(batch["suffix"][i])
    # 将数据转换为llama支持的格式
    if not only_prefix:
        context = ["<SUFFIX>" + s + "<PREFIX>" + p + "<MIDDLE>" for p, s in zip(valid_batch["prefix"], valid_batch["suffix"])]
    else:
        context = ["<SUFFIX><PREFIX>" + p + "<MIDDLE>" for p in valid_batch["prefix"]]
    generated_results = llm.generate(context, sampling_params, use_tqdm=False)
    generated_code = [[output.text for output in generated_result.outputs] for generated_result in generated_results]
    valid_batch["prediction"] = generated_code
    return valid_batch

def process_dataset(dataset:datasets.Dataset, model_path:str, idx:int, only_prefix:bool) -> datasets.Dataset:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=100, include_stop_str_in_output=True)
    if "deepseek" in model_path:
        batch_func = partial(deepseek_process_batch, llm=llm, sampling_params=sampling_params, only_prefix=only_prefix)
        batch_size = 64
    elif "llama" in model_path:
        batch_func = partial(llama_process_batch, llm=llm, sampling_params=sampling_params, only_prefix=only_prefix)
        batch_size = 128
    else:
        raise Exception("Unsupport model.")
    remove_columns = []
    if "input_ids" in dataset.column_names:
        remove_columns.append("input_ids")
    if "attention_mask" in dataset.column_names:
        remove_columns.append("attention_mask")
    dataset = dataset.map(batch_func, batched=True, batch_size=batch_size, num_proc=1, remove_columns=remove_columns)
    return dataset

# 判断预测值是否与reference足够相似
def satisify(prediction:str, reference:str):
    pre_lines = prediction.rstrip().split('\n')
    pre_lines = [line + '\n' if i != len(pre_lines) else line for i, line in enumerate(pre_lines)]
    ref_lines = reference.rstrip().split('\n')
    ref_lines = [line + '\n' if i != len(ref_lines) else line for i, line in enumerate(ref_lines)]
    
    for pre, ref in zip(pre_lines, ref_lines):
        if pre != ref:
            return False
    return True

def filter_batch(batch, correct:bool=True):
    valid_batch = {"prefix":[], "suffix":[], "middle":[]}
    for i, (m, p) in enumerate(zip(batch["middle"], batch["prediction"])):
        if (correct and satisify(p[0], m)) or (not correct and not satisify(p[0], m)):
            valid_batch["prefix"].append(batch["prefix"][i])
            valid_batch["middle"].append(batch["middle"][i])
            valid_batch["suffix"].append(batch["suffix"][i])
            
    return valid_batch

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to the raw dataset.")
    parser.add_argument("--output_data_path", type=str, required=True, help="Path to save the processed dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--num_shards", type=int, required=True, help="Number of shards to split the dataset into.")
    parser.add_argument("--only_prefix", action="store_true")
    
    args = parser.parse_args()
    
    raw_data_path = args.raw_data_path
    output_data_path = args.output_data_path
    model_path = args.model_path
    num_shards = args.num_shards
    only_prefix:bool = args.only_prefix
    
    dataset = datasets.load_from_disk(raw_data_path)
    shards = [dataset.shard(num_shards, i, contiguous=False) for i in range(num_shards)]

    with ProcessPoolExecutor(max_workers=num_shards) as executor:
        futures = [executor.submit(process_dataset, shard, model_path, i, only_prefix) for i, shard in enumerate(shards)]
        
        filter_datasets = []
        for future in as_completed(futures):
            res = future.result()
            filter_datasets.append(res)

    merged_dataset:datasets.Dataset = datasets.concatenate_datasets(filter_datasets)

    if "deepseek" in model_path:
        filter_func = filter_batch
    elif "llama" in model_path:
        filter_func = partial(filter_batch, correct=False)
    output_dataset = merged_dataset.map(filter_func, batched=True, batch_size=64, num_proc=os.cpu_count(), remove_columns=["prediction"])
    output_dataset.save_to_disk(output_data_path)
    
if __name__ == "__main__":
    main()