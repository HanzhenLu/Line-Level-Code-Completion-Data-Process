import random
import re
import tokenize
import io
import argparse
import os
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Dict, Optional, Union
from datasets import DatasetDict, load_from_disk, load_dataset
from functools import partial

class Group:
    def __init__(self, valid:bool=True, code_snippet:str = "", \
        tokens:Optional[List[int]] = None, mask:Optional[List[int]] = None) -> None:
        self.valid = valid
        self.code_snippet:str = code_snippet
        self.tokens:Optional[List[int]] = tokens
        self.mask:Optional[List[int]] = mask

def split_and_tokenize(tokenizer: "PreTrainedTokenizer", code:str, keywords_list:List[str]) -> Dict[str, List[int]]:
    tokenized_results = []
    jump_table = {}
    # 防止将<UNK>等特殊字符切割开来
    for keyword in keywords_list:
        indices = [match.start() for match in re.finditer(re.escape(keyword), code)]
        for indice in indices:
            for i in range(1, len(keyword)):
                jump_table[indice + i] = len(keyword) - i
                
    cur_pos = 0
    while True:
        step = random.randint(1, 100)
        if step + cur_pos >= len(code):
            tokenized_results.append(tokenizer(code[cur_pos:], add_special_tokens=False))
            break
        else:
            if cur_pos + step in jump_table:
                step += jump_table[cur_pos + step]
            tokenized_results.append(tokenizer(code[cur_pos:cur_pos+step], add_special_tokens=False))
            cur_pos += step
    
    return {
        "input_ids": [id for snippet in tokenized_results for id in snippet["input_ids"]],
        "attention_mask": [mask for snippet in tokenized_results for mask in snippet["attention_mask"]]
    }
    
def segment(res: Dict[str, List[int]], cutoff_len: int) -> List[Dict[str, List[int]]]:
    input_ids_chunks = []
    attention_mask_chunks = []
    
    # 计算每次切片的起始位置
    for i in range(0, len(res["input_ids"]), cutoff_len):
        # 确保切片不会超出列表范围
        if i + cutoff_len <= len(res["input_ids"]):
            input_ids_chunks.append(res["input_ids"][i:i + cutoff_len])
            attention_mask_chunks.append(res["attention_mask"][i:i + cutoff_len])
        else:
            # 处理最后一个切片，确保不会超出范围
            input_ids_chunks.append(res["input_ids"][-cutoff_len:])
            attention_mask_chunks.append(res["attention_mask"][-cutoff_len:])
            break
    
    return [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(input_ids_chunks, attention_mask_chunks)]

def map_line(code:str) -> List[List[int]]:
    stack = []
    line_map = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    
    # start:Tuple[int, int]，代表着起始的行和列，这里的行是从1开始的
    for token_type, string, start, _, _ in tokens:
        # OP
        if token_type == 54:
            if string == '{' or string == '[' or string == '(':
                stack.append(string)
            elif string == '}' or string == ']' or string == ')':
                stack.pop()
                    
        # NL
        if token_type == 61 and len(stack) == 0:
            line_map.append([start[0] - 1])
            line_count = start[0]
        
        # NEWLINE
        elif token_type == 4:
            line_map.append([i - 1 for i in range(line_count+1, start[0]+1)])
            line_count = start[0]
    
    return line_map

def code2ids(tokenizer: "PreTrainedTokenizer", text_examples: List[str], data_args:argparse.Namespace) -> Dict[str, List[Union[str, List[int]]]]:
    # 需要预留一些token出来
    cutoff_len = data_args.cutoff_len - 10
    
    def NTP(code:str) -> List[Dict[str, List[int]]]:
        samples = []
        res = split_and_tokenize(tokenizer, code, ["<EOS>", "<BOS>", "<UNK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<PAD>", "<MASK>"])
        if len(res["input_ids"]) > cutoff_len:
            samples.extend(segment(res, cutoff_len))
        else:
            samples.append(res)
        return samples
    
    def char_level_FIM(block:List[Group]) -> Optional[Dict[str, Union[List[int], str]]]:
        # 挑选出适合作为middle的tokens_group
        # 不能是空行，不能太长也不能太短
        # 不能是边界处的语句
        valid_group_ids = [i for i, group in enumerate(block) 
                           if not group.code_snippet.isspace() and group.code_snippet != "" 
                           and len(group.code_snippet) < 400 and len(group.code_snippet) > 2
                           and group.valid]
        if len(valid_group_ids) != 0:
            if len(valid_group_ids) == 1:
                selected_group_id = valid_group_ids[0]
            else:
                selected_group_id = valid_group_ids[random.randint(0, len(valid_group_ids) -1)]
            selected_group = block[selected_group_id]
            position = random.randint(0, len(selected_group.code_snippet) - 2)
            prefix = selected_group.code_snippet[:position]
            middle_code = selected_group.code_snippet[position:]
            prefix_code = "".join(group.code_snippet for group in block[:selected_group_id]) + prefix
            suffix_code = "".join(group.code_snippet for group in block[selected_group_id+1:])
            
            # PSM
            if random.random() > 0.5:
                code = "<PREFIX>" + prefix_code + "<SUFFIX>" + suffix_code + "<MIDDLE>" + middle_code
                                
            # SPM
            else:
                code = "<PREFIX><SUFFIX>" + suffix_code + "<MIDDLE>" + prefix_code + middle_code
            
            res = split_and_tokenize(tokenizer, code, ["<EOS>", "<BOS>", "<UNK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<PAD>", "<MASK>"])
                    
            return {
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"],
                "prefix": prefix_code,
                "middle": middle_code,
                "suffix": suffix_code
            } if data_args.save_code else {
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"]
            }
            
        else:
            return None
    
    # BPE dropoout
    # tokenizer._tokenizer.model.dropout = 0.1
    
    final_samples:List[Dict[str, Union[str, List[int]]]] = []
    for text in text_examples:
        if data_args.do_FIM:
            
            # 10%的概率进行NTP
            if random.random() > 0.9:
                final_samples.extend(NTP(text))
                continue
                
            line_map = map_line(text.replace("<UNK>", "zxcv"))
            raw_lines = text.split("\n")
            raw_lines = [line+"\n" if i != len(raw_lines) - 1 else line for i, line in enumerate(raw_lines)]
            line_group:List[Group] = []
            for line_idx in line_map:
                line_group.append(Group(True, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
            
            blocks:List[List[Group]] = []
            block:List[Group] = []
            cur_length = 0
            for group in line_group:
                res = tokenizer(group.code_snippet, add_special_tokens=False)
                
                # 一个语句就能塞满整个上下文窗口的话，这种数据不要也罢
                if len(res["input_ids"]) > cutoff_len:
                    continue
                
                group.tokens = res["input_ids"]
                group.mask = res["attention_mask"]
                if cur_length + len(group.tokens) > cutoff_len:
                    # 将边界处的tokens group拆分成两个，
                    # 一个是归到下一个sample的next_group，一个是属于当前sample的group
                    next_group = Group(False, "",
                        res["input_ids"][cutoff_len-cur_length-len(group.tokens):],
                        res["attention_mask"][cutoff_len-cur_length-len(group.mask):])
                    next_group.code_snippet = tokenizer.decode(next_group.tokens)
                    
                    group.valid = False
                    group.tokens = res["input_ids"][:cutoff_len-cur_length-len(group.tokens)]
                    group.mask = res["attention_mask"][:cutoff_len-cur_length-len(group.mask)]
                    group.code_snippet = group.code_snippet[:-len(next_group.code_snippet)]
                    
                    block.append(group)
                    result = char_level_FIM(block)
                    if result is not None:
                        final_samples.append(result)
                    
                    blocks.append(block)
                    cur_length = len(next_group.tokens)
                    block = [next_group]
                    
                elif cur_length + len(group.tokens) == cutoff_len:
                    block.append(group)
                    result = char_level_FIM(block)
                    if result is not None:
                        final_samples.append(result)
                    
                    blocks.append(block)
                    cur_length = 0
                    block = []
                
                else:
                    block.append(group)
                    cur_length += len(group.tokens)
                    
            if len(blocks) != 0 and len(block) > 0:
                for group in reversed(blocks[-1]):
                    if cur_length + len(group.tokens) >= cutoff_len:
                        previous_group = Group(False, "", 
                            group.tokens[cur_length-cutoff_len:],
                            group.mask[cur_length-cutoff_len:])
                        previous_group.code_snippet = tokenizer.decode(previous_group.tokens)
                        block = [previous_group] + block
                        result = char_level_FIM(block)
                        if result is not None:
                            final_samples.append(result)
                        break
                    else:
                        block = [group] + block
                        cur_length += len(group.tokens)
            
            elif len(blocks) == 0 and len(block) > 0:
                result = char_level_FIM(block)
                if result is not None:
                    final_samples.append(result)
            
        else:
            final_samples.extend(NTP(text))
    
    input_ids, attention_mask, prefix, middle, suffix = [], [], [], [], []
    for sample in final_samples:
        input_ids.append(sample["input_ids"] + [tokenizer.eos_token_id])
        attention_mask.append(sample["attention_mask"] + [1])
        if "prefix" in sample:
            prefix.append(sample["prefix"])
            middle.append(sample["middle"])
            suffix.append(sample["suffix"])
        else:
            prefix.append("")
            middle.append("")
            suffix.append("")
    
    return {
        "input_ids":input_ids, 
        "attention_mask":attention_mask,
        "prefix":prefix,
        "middle":middle,
        "suffix":suffix
    } if data_args.save_code else {
        "input_ids":input_ids, 
        "attention_mask":attention_mask
    }

def process_batch(batch, process_func):
    res = process_func(text_examples=batch["text"])
    for key, value in res.items():
        batch[key] = value
    return batch

def tokenize_batch(batch, tokenizer:AutoTokenizer):
    return_batch = {"input_ids":[], "attention_mask":[]}
    for code in batch["text"]:
        res = split_and_tokenize(tokenizer, code, ["<EOS>", "<BOS>", "<UNK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<PAD>", "<MASK>"])
        return_batch["input_ids"].append(res["input_ids"] + [tokenizer.eos_token_id])
        return_batch["attention_mask"].append(res["attention_mask"] + [1])
    return return_batch

def main():
    parser = argparse.ArgumentParser("Preprocess and tokenize datasets for training.")
    
    parser.add_argument("--datasets_path", help="specify the directory containing the raw datasets to be processed.", required=True)
    parser.add_argument("--output_path", help="specify the directory where the processed datasets will be saved.", required=True)
    parser.add_argument("--do_FIM", action="store_true", help="create a dataset for filling in the middle or for next token prediction")
    parser.add_argument("--save_code", action="store_true", help="whether to save the prefix, middle and suffix")
    parser.add_argument("--only_tokenize", action="store_true", help="indicates that the data has already been preprocessed and only needs to be tokenized.")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    
    data_args = parser.parse_args()
    dataset_path = data_args.datasets_path
    
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_bbpe_keywords")
    
    data_files = []
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        data_files.append(file_path)
    
    if data_args.max_workers is None:
        max_workers = min(len(data_files), os.cpu_count())
    else:
        max_workers = data_args.max_workers
    
    if os.path.join(dataset_path, "dataset_info.json") not in data_files:
        dataset = load_dataset(
                "arrow",
                split="train",
                data_files=data_files,
                num_proc=os.cpu_count(),
            )
    else:
        dataset = load_from_disk(dataset_path)
        
    if "blob_id" in dataset.column_names:
        remove_columns = ["text", "blob_id"]
    else:
        remove_columns = ["text"]
    
    if data_args.only_tokenize:
        batch_func = partial(tokenize_batch, tokenizer=tokenizer)
    else:
        process_func = partial(code2ids, tokenizer=tokenizer, data_args=data_args)
        batch_func = partial(process_batch, process_func=process_func)
    
    dataset = dataset.map(batch_func, batched=True, batch_size=40, num_proc=max_workers, remove_columns=remove_columns)
    dataset = dataset.train_test_split(test_size=data_args.val_size)
    dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    dataset.save_to_disk(data_args.output_path)


if __name__ == "__main__":
    main()