import random
import re
import tokenize
import io
import argparse
import os
import ast
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
        
    def get_code_line_count(self) -> int:
        """计算 code_snippet 的行数"""
        return self.code_snippet.count('\n') + 1 if self.code_snippet else 0

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
    '''
    将代码块按逻辑行分割开来
    '''
    stack = []
    line_map = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    
    # start:Tuple[int, int]，代表着起始的行和列，这里的行是从1开始的
    for token_type, string, start, _, _ in tokens:
        # OP
        if token_type == tokenize.OP:
            if string == '{' or string == '[' or string == '(':
                stack.append(string)
            elif string == '}' or string == ']' or string == ')':
                stack.pop()
                    
        # NL仅代表换行，不代表语句的结尾
        # 但是由于注释所在行的换行被标注为NL，为了防止注释被划分到下一个语句中，所以要加一个特殊判断
        if token_type == tokenize.NL and len(stack) == 0:
            line_map.append([start[0] - 1])
            line_count = start[0]
        
        # NEWLINE
        elif token_type == tokenize.NEWLINE:
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
    
    def line_level_FIM(block:List[Group]) -> Optional[Dict[str, Union[List[int], str]]]:
        # 挑选出适合作为middle的tokens_group
        # 不能是空行，不能太长也不能太短
        # 不能是边界处的语句
        valid_group_ids = [i for i, group in enumerate(block) 
                           if not group.code_snippet.isspace() and group.code_snippet != "" 
                           and len(group.code_snippet) < 400 and len(group.code_snippet.strip()) > 10
                           and group.valid]
        if len(valid_group_ids) != 0:
            if len(valid_group_ids) == 1:
                selected_group_id = valid_group_ids[0]
            else:
                weights = [(group.get_code_line_count() + 1) / 2 for group in [block[i] for i in valid_group_ids]]
                selected_group_id = random.choices(valid_group_ids, weights=weights, k=1)[0]
            selected_group = block[selected_group_id]
            
            # 清除行末的空格
            selected_lines = [line.rstrip() + "\n" for line in selected_group.code_snippet.splitlines()]
            selected_group.code_snippet = "".join(selected_lines)
            
            # 根据腾讯给出的数据，只有20%的情况是从行首开始补全的
            if random.random() > 0.2:
                start = random.randint(0, len(selected_group.code_snippet) - 5)
            else:
                start = 0
            if selected_group.code_snippet.count("\n") > 1 and random.random() > 0.5:
                newline_indices = [i for i, char in enumerate(selected_group.code_snippet) if char == "\n" and i > start]
                if len(newline_indices) > 0:
                    end = random.choice(newline_indices) + 1
                else:
                    end = len(selected_group.code_snippet)
            else:
                end = len(selected_group.code_snippet)
                    
            prefix = selected_group.code_snippet[:start]
            middle_code = selected_group.code_snippet[start:end]
            
            suffix = selected_group.code_snippet[end:]
            prefix_code = "".join(group.code_snippet for group in block[:selected_group_id]) + prefix
            suffix_code = suffix + "".join(group.code_snippet for group in block[selected_group_id+1:])
            
            if data_args.only_PSM:
                threshold = 0.0
            elif data_args.only_SPM:
                threshold = 1.0
            else:
                threshold = 0.5
            # PSM
            if random.random() > threshold:
                code = "<PREFIX>" + prefix_code + "<SUFFIX>" + suffix_code + "<MIDDLE>" + middle_code
                                
            # SPM
            else:
                code = "<SUFFIX>" + suffix_code + "<PREFIX>" + prefix_code + "<MIDDLE>" + middle_code
            
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
    
    def func_level_FIM(code: str) -> Optional[Dict[str, str]]:
        """
        分析输入的Python代码，拆分函数定义部分。
        
        参数:
            code: 要分析的Python代码字符串
            
        返回:
            如果代码中没有函数定义，返回None。
            否则返回包含'prefix', 'middle', 'suffix'的字典。
        """
        try:
            # 解析代码为AST
            tree = ast.parse(code)
        except SyntaxError:
            return None
        
        # 收集所有函数定义节点
        function_defs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and 3 <= node.end_lineno - node.lineno <= 30
        ]
        
        if not function_defs:
            return None
        
        # 随机选择一个函数定义
        selected_func = random.choice(function_defs)
        
        # 获取函数在源代码中的行范围
        func_lines = code.splitlines(keepends=True)
        func_start = selected_func.lineno - 1  # AST行号从1开始
        func_end = selected_func.end_lineno  # end_lineno是包含的
        
        # 获取函数签名和docstring部分
        # 函数体开始于第一个非docstring、非空的行
        body_start = func_start
        if selected_func.body and isinstance(selected_func.body[0], ast.Expr) and \
        isinstance(selected_func.body[0].value, ast.Str):
            # 有docstring的情况，跳过docstring
            docstring_end = selected_func.body[0].end_lineno
            body_start = docstring_end
        elif selected_func.body:
            body_start = selected_func.body[0].lineno - 1
        
        # 拆分prefix和middle
        prefix_lines = func_lines[:body_start]
        
        # 函数体部分
        body_lines = func_lines[body_start:func_end]
        
        # 组合prefix
        prefix = ''.join(prefix_lines)
        
        # middle部分
        middle = ''.join(body_lines)
        
        # suffix部分是函数定义之后的所有代码
        suffix = ''.join(func_lines[func_end:])
        
        assert prefix + middle + suffix == code
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        middle_ids = tokenizer(middle, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        if len(middle_ids) > data_args.cutoff_len // 4:
            return None
        left_budget = data_args.cutoff_len - len(middle_ids) - 4
        prefix_budget = left_budget // 2
        suffix_budget = left_budget - prefix_budget
        if len(prefix_ids) < prefix_budget:
            suffix_ids = suffix_ids[:left_budget - len(prefix_ids)]
        elif len(suffix_ids) < suffix_budget:
            prefix_ids = prefix_ids[-(left_budget - len(suffix_ids)):]
        else:
            prefix_ids = prefix_ids[-prefix_budget:]
            suffix_ids = suffix_ids[:suffix_budget]
        
        if data_args.only_PSM:
                threshold = 0.0
        elif data_args.only_SPM:
            threshold = 1.0
        else:
            threshold = 0.5
        # PSM
        if random.random() > threshold:
            input_ids = [tokenizer.convert_tokens_to_ids("<PREFIX>")] + prefix_ids + [tokenizer.convert_tokens_to_ids("<SUFFIX>")] + \
                suffix_ids + [tokenizer.convert_tokens_to_ids("<MIDDLE>")] + middle_ids         
        # SPM
        else:
            input_ids = [tokenizer.convert_tokens_to_ids("<SUFFIX>")] + suffix_ids + [tokenizer.convert_tokens_to_ids("<PREFIX>")] + \
                prefix_ids + [tokenizer.convert_tokens_to_ids("<MIDDLE>")] + middle_ids     
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            'prefix': tokenizer.decode(prefix_ids, skip_special_tokens=False),
            'middle': tokenizer.decode(middle_ids, skip_special_tokens=False),
            'suffix': tokenizer.decode(suffix_ids, skip_special_tokens=False),
        }
    
    def multi_lines_FIM(block:List[Group]):
        merged_groups = []
        current_snippet = []
        current_tokens = []
        current_mask = []
        current_line_count = 0
        
        for group in block:
            line_count = group.get_code_line_count()
            if current_line_count + line_count <= 10:
                current_snippet.append(group.code_snippet)
                current_tokens.extend(group.tokens or [])
                current_mask.extend(group.mask or [])
                current_line_count += line_count
            else:
                # 保存当前合并的Group
                if current_snippet:
                    merged_groups.append(Group(
                        valid=True,
                        code_snippet="".join(current_snippet),
                        tokens=current_tokens if current_tokens else None,
                        mask=current_mask if current_mask else None
                    ))
                
                # 开始新的合并组
                current_snippet = [group.code_snippet]
                current_tokens = group.tokens.copy() if group.tokens else []
                current_mask = group.mask.copy() if group.mask else []
                current_line_count = line_count
        
        if current_snippet:
            merged_groups.append(Group(
                valid=True,
                code_snippet="".join(current_snippet),
                tokens=current_tokens if current_tokens else None,
                mask=current_mask if current_mask else None
            ))
        
        selected_group = random.choices(merged_groups, k=1)[0]
        
        # 清除行末的空格
        selected_lines = [line.rstrip() + "\n" for line in selected_group.code_snippet.splitlines()]
        selected_group.code_snippet = "".join(selected_lines)
        
        if random.random() > 0.2:
            if len(selected_group.code_snippet).strip() < 5:
                return None
            start = random.randint(0, len(selected_group.code_snippet) - 5)
        else:
            start = 0
        
        prefix = selected_group.code_snippet[:start]
        middle_code = selected_group.code_snippet[start:]
        
        selected_group_id = merged_groups.index(selected_group)
        prefix_code = "".join(group.code_snippet for group in block[:selected_group_id]) + prefix
        suffix_code = "".join(group.code_snippet for group in block[selected_group_id+1:])
        
        if data_args.only_PSM:
            threshold = 0.0
        elif data_args.only_SPM:
            threshold = 1.0
        else:
            threshold = 0.5
        # PSM
        if random.random() > threshold:
            code = "<PREFIX>" + prefix_code + "<SUFFIX>" + suffix_code + "<MIDDLE>" + middle_code
                            
        # SPM
        else:
            code = "<SUFFIX>" + suffix_code + "<PREFIX>" + prefix_code + "<MIDDLE>" + middle_code
        
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
    
    # BPE dropoout
    # tokenizer._tokenizer.model.dropout = 0.1
    
    final_samples:List[Dict[str, Union[str, List[int]]]] = []
    for text in text_examples:
        if data_args.do_FIM:
            
            # 10%的概率进行NTP
            random_num = random.random()
            if random_num > 0.9:
                final_samples.extend(NTP(text))
                continue
            elif random_num > 0.7 and data_args.do_func_level_fim:
                result = func_level_FIM(text)
                if result:
                    final_samples.append(result)
            elif random_num > 0.5 and data_args.do_multi_lines_fim:
                fim_func = multi_lines_FIM
            else:
                fim_func = line_level_FIM
            
            try:
                line_map = map_line(text.replace("<UNK>", "zxcv"))
            except:
                continue
            raw_lines = text.splitlines(keepends=True)
            line_group:List[Group] = []
            for line_idx in line_map:
                # 当 ignore_comment 被使用时，禁止以#，"""和'''开头的语句被选作middle
                # 因为它们大概率是注释
                if data_args.ignore_comment:
                    code_snippet = "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])
                    if code_snippet.strip().startswith("#") or code_snippet.strip().startswith("'''") or code_snippet.strip().startswith('"""'):
                        line_group.append(Group(False, code_snippet))
                    else:
                        line_group.append(Group(True, code_snippet))
                else:
                    line_group.append(Group(True, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
            
            blocks:List[List[Group]] = []
            block:List[Group] = []
            cur_length = 0
            for group in line_group:
                res = tokenizer(group.code_snippet, add_special_tokens=False)
                
                # 一个语句就能塞满整个上下文窗口的话，这种数据不要
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
                    result = fim_func(block)
                    if result is not None:
                        final_samples.append(result)
                    
                    blocks.append(block)
                    cur_length = len(next_group.tokens)
                    block = [next_group]
                    
                elif cur_length + len(group.tokens) == cutoff_len:
                    block.append(group)
                    result = fim_func(block)
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
                        result = fim_func(block)
                        if result is not None:
                            final_samples.append(result)
                        break
                    else:
                        block = [group] + block
                        cur_length += len(group.tokens)
            
            elif len(blocks) == 0 and len(block) > 0:
                result = fim_func(block)
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
    
    parser.add_argument("--datasets_path", help="Path to the directory containing the raw datasets to be processed.", required=True)
    parser.add_argument("--output_path", help="Path to the directory where the processed datasets will be saved.", required=True)
    parser.add_argument("--do_FIM", action="store_true", help="Enable this flag to create a dataset for Fill-in-the-Middle (FIM) or next token prediction tasks.")
    parser.add_argument("--save_code", action="store_true", help="Enable this flag to save the prefix, middle, and suffix separately in the processed dataset.")
    parser.add_argument("--only_tokenize", action="store_true", help="Enable this flag if the data has already been preprocessed and only requires tokenization.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the dataset to be used as the validation set (default: 0.1).")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker processes to use for data processing (default: None, which uses all available cores).")
    parser.add_argument("--cutoff_len", type=int, default=2048, help="Maximum number of tokens allowed in a single sample (default: 2048).")
    parser.add_argument("--ignore_comment", action="store_true", help="Enable this flag to forbid comment be chosen as middle")
    parser.add_argument("--only_PSM", action="store_true", help="Enable this flag to use PSM only")
    parser.add_argument("--only_SPM", action="store_true", help="Enable this flag to use PSM only")
    parser.add_argument("--do_func_level_fim", action="store_true")
    parser.add_argument("--sampled_ratio", type=float, default=None)
    
    data_args = parser.parse_args()
    dataset_path = data_args.datasets_path
    assert not (data_args.only_PSM and data_args.only_SPM)
    
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
        
    if data_args.sampled_ratio:
        shuffled_dataset = dataset.shuffle(seed=42)
        dataset = shuffled_dataset.select(range(int(data_args.sampled_ratio * len(shuffled_dataset))))
    
    dataset = dataset.map(batch_func, batched=True, batch_size=40, num_proc=max_workers, remove_columns=remove_columns)
    dataset = dataset.train_test_split(test_size=data_args.val_size)
    dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    dataset.save_to_disk(data_args.output_path)


if __name__ == "__main__":
    main()