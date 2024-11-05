import os
import datasets
import random
from vllm import LLM, SamplingParams
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

raw_data_path = "/data/hanzhenlu/dataset/Python-deduped/deduplicated_tokenized"
output_data_path = "/data/hanzhenlu/DataProcess/high-quality"
model_path = "/nasdata/Model/deepseek-coder-6.7b-base"
num_shards = 8
def process_batch(batch, llm:LLM, sampling_params:SamplingParams):
    valid_batch = {"prefix":[], "suffix":[], "middle":[]}
    for i, m in enumerate(batch["middle"]):
        if m != "":
            valid_batch["prefix"].append(batch["prefix"][i])
            valid_batch["middle"].append(batch["middle"][i])
            valid_batch["suffix"].append(batch["suffix"][i])
    context = ["<｜fim▁begin｜>" + p + "<｜fim▁hole｜>" + s + "<｜fim▁end｜>" for p, s in zip(valid_batch["prefix"], valid_batch["suffix"])]
    generated_results = llm.generate(context, sampling_params, use_tqdm=False)
    generated_code = [[output.text for output in generated_result.outputs] for generated_result in generated_results]
    valid_batch["prediction"] = generated_code
    return valid_batch

def process_dataset(dataset:datasets.Dataset, idx:int) -> datasets.Dataset:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(n=1, temperature=0.8, top_p=0.95, max_tokens=100, include_stop_str_in_output=True)
    batch_func = partial(process_batch, llm=llm, sampling_params=sampling_params)
    dataset = dataset.map(batch_func, batched=True, batch_size=64, num_proc=1, remove_columns=["input_ids", "attention_mask"])
    return dataset

def satisify(prediction:str, reference:str):
    pre_lines = prediction.rstrip().split('\n')
    pre_lines = [line + '\n' if i != len(pre_lines) else line for i, line in enumerate(pre_lines)]
    ref_lines = reference.rstrip().split('\n')
    ref_lines = [line + '\n' if i != len(ref_lines) else line for i, line in enumerate(ref_lines)]
    
    for pre, ref in zip(pre_lines, ref_lines):
        if pre != ref:
            return False
    return True

def filter_batch(batch):
    valid_batch = {"prefix":[], "suffix":[], "middle":[]}
    for i, (m, p) in enumerate(zip(batch["middle"], batch["prediction"])):
        if satisify(p[0], m):
            valid_batch["prefix"].append(batch["prefix"][i])
            valid_batch["middle"].append(batch["middle"][i])
            valid_batch["suffix"].append(batch["suffix"][i])
    return valid_batch

dataset = datasets.load_from_disk(raw_data_path)
# merged_dataset:datasets.Dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"]])
sample_size = int(len(dataset["train"]) * 0.2)

sampled_indices = random.sample(range(len(dataset["train"])), sample_size)
sampled_dataset = dataset["train"].select(sampled_indices)
shards = [sampled_dataset.shard(num_shards, i, contiguous=False) for i in range(num_shards)]

with ProcessPoolExecutor(max_workers=num_shards) as executor:
    futures = [executor.submit(process_dataset, shard, i) for i, shard in enumerate(shards)]
    
    filter_datasets = []
    for future in as_completed(futures):
        res = future.result()
        filter_datasets.append(res)

merged_dataset:datasets.Dataset = datasets.concatenate_datasets(filter_datasets)

output_dataset = merged_dataset.map(filter_batch, batched=True, batch_size=64, num_proc=os.cpu_count())
output_dataset.save_to_disk(output_data_path)