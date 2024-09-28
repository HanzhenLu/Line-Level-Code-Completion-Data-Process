import random
import tokenize
import io
import argparse
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, Optional, List, Callable

# def split_tokens(res: Dict[str, List[int]], cutoff_len: int) -> List[Dict[str, List[int]]]:
#     input_ids_chunks = []
#     attention_mask_chunks = []
    
#     # 计算每次切片的起始位置
#     for i in range(0, len(res["input_ids"]), cutoff_len):
#         # 确保切片不会超出列表范围
#         if i + cutoff_len <= len(res["input_ids"]):
#             input_ids_chunks.append(res["input_ids"][i:i + cutoff_len])
#             attention_mask_chunks.append(res["attention_mask"][i:i + cutoff_len])
#         else:
#             # 处理最后一个切片，确保不会超出范围
#             input_ids_chunks.append(res["input_ids"][-cutoff_len:])
#             attention_mask_chunks.append(res["attention_mask"][-cutoff_len:])
#             break
    
#     return [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(input_ids_chunks, attention_mask_chunks)]

# def NTP(tokenizer:PreTrainedTokenizer, code:str, cutoff_len:int) -> List[Dict[str, List[int]]]:
#     samples = []
#     res = tokenizer(code, add_special_tokens=False)
#     if len(res["input_ids"]) > cutoff_len:
#         samples.extend(split_tokens(res, cutoff_len))
#     else:
#         samples.append(res)
#     return samples

# class Group:
#     def __init__(self, valid:bool = False, code_snippet:str = "", \
#         tokens:Optional[List[int]] = None, mask:Optional[List[int]] = None) -> None:
#         self.valid:bool = valid
#         self.code_snippet:str = code_snippet
#         self.tokens:Optional[List[int]] = tokens
#         self.mask:Optional[List[int]] = mask
        
# def map_line(code:str) -> List[Tuple[bool, List[int]]]:
#     stack = []
#     line_map = []
#     line_count = 0
#     tokens = tokenize.generate_tokens(io.StringIO(code).readline)
#     has_op = False
#     has_return = False
#     has_import = False
#     for token_type, string, start, _, _ in tokens:
#         # OP
#         if token_type == 54:
#             has_op = True
#             if string == '{' or string == '[' or string == '(':
#                 stack.append(string)
#             elif string == '}' or string == ']' or string == ')':
#                 stack.pop()
                    
#         # NL
#         if token_type == 61 and len(stack) == 0:
#             line_map.append((False, [start[0] - 1]))
#             line_count = start[0]
        
#         # NEWLINE
#         # has_op 是为了防止长字符串被当作middle
#         # has_return 是因为return不会被当作op
#         # has_import 同理
#         elif token_type == 4:
#             if has_op or has_return or has_import:
#                 line_map.append((True, [i - 1 for i in range(line_count+1, start[0]+1)]))
#             else:
#                 line_map.append((False, [i - 1 for i in range(line_count+1, start[0]+1)]))
#             line_count = start[0]
#             has_op = False
#             has_return = False
#             has_import = False
        
#         # NAME
#         elif token_type == 1:
#             if string == "return":
#                 has_return = True
#             if string == "import":
#                 has_import = True
    
#     return line_map

# def char_level_FIM(tokenizer: "PreTrainedTokenizer", text_examples: List[str], cutoff_len: int) -> List[str]:
#     # 需要预留一些token出来
#     cutoff_len -= 10
    
#     def FIM(block:List[Group]) -> str:
#         valid_group_ids = [i for i, group in enumerate(block) if group.valid]
#         if len(valid_group_ids) != 0:
#             if len(valid_group_ids) == 1:
#                 selected_group_id = valid_group_ids[0]
#             else:
#                 selected_group_id = valid_group_ids[random.randint(0, len(valid_group_ids) -1)]
#             selected_group = block[selected_group_id]
#             position = random.randint(0, len(selected_group.code_snippet))
#             prefix = selected_group.code_snippet[:position]
#             middle_code = selected_group.code_snippet[position:]
#             prefix_code = "".join(group.code_snippet for group in block[:selected_group_id]) + prefix
#             suffix_code = "".join(group.code_snippet for group in block[selected_group_id+1:])
            
#             # PSM
#             if random.random() > 0.5:
#                 code = "<PREFIX>" + prefix_code + "<SUFFIX>" + suffix_code + "<MIDDLE>" + middle_code
                                
#             # SPM
#             else:
#                 code = "<PREFIX><SUFFIX>" + suffix_code + "<MIDDLE>" + prefix_code + middle_code
                    
#             return code

#         else:
#             return None
    
#     end_id = tokenizer.eos_token_id
    
#     final_samples:List[Dict[str, List[int]]] = []
#     for text in text_examples:
#         if random.random() < 0.9:
#             line_map = map_line(text.replace("<UNK>", "zxcv"))
#             raw_lines = text.split("\n")
#             raw_lines = [line+"\n" if i != len(raw_lines) - 1 else line for i, line in enumerate(raw_lines)]
#             line_group:List[Group] = []
#             for valid, line_idx in line_map:
#                 if len(line_group) == 0:
#                     line_group.append(Group(valid, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
#                 else:
#                     if not valid and valid == line_group[-1].valid:
#                         line_group[-1].code_snippet += "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])
#                     else:
#                         line_group.append(Group(valid, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
            
#             blocks:List[List[Group]] = []
#             block:List[Group] = []
#             cur_length = 0
            
#             for group in line_group:
#                 res = tokenizer(group.code_snippet, add_special_tokens=False)
                
#                 # 一个语句就能塞满整个上下文窗口的话，这种数据不要也罢
#                 if len(res["input_ids"]) > cutoff_len:
#                     continue
                
#                 group.tokens = res["input_ids"]
#                 group.mask = res["attention_mask"]
#                 if cur_length + len(group.tokens) > cutoff_len:
#                     next_group = Group(False, "",
#                         res["input_ids"][cutoff_len-cur_length-len(group.tokens):],
#                         res["attention_mask"][cutoff_len-cur_length-len(group.mask):])
#                     next_group.code_snippet = tokenizer.decode(next_group.tokens)
#                     group.valid = False
#                     group.tokens = res["input_ids"][:cutoff_len-cur_length-len(group.tokens)]
#                     group.mask = res["attention_mask"][:cutoff_len-cur_length-len(group.mask)]
#                     group.code_snippet = group.code_snippet[:-len(next_group.code_snippet)]
#                     block.append(group)
#                     result = FIM(block)
#                     if result is not None:
#                         final_samples.append(result)
                    
#                     blocks.append(block)
#                     cur_length = len(next_group.tokens)
#                     block = [next_group]
#                 elif cur_length + len(group.tokens) == cutoff_len:
#                     block.append(group)
#                     result = FIM(block)
#                     if result is not None:
#                         final_samples.append(result)
                    
#                     blocks.append(block)
#                     cur_length = 0
#                     block = []
#                 else:
#                     block.append(group)
#                     cur_length += len(group.tokens)
                    
#             if len(blocks) != 0 and len(block) > 0:
#                 for group in reversed(blocks[-1]):
#                     if cur_length + len(group.tokens) >= cutoff_len:
#                         previous_group = Group(False, "", 
#                             group.tokens[cur_length-cutoff_len:],
#                             group.tokens[cur_length-cutoff_len:])
#                         previous_group.code_snippet = tokenizer.decode(previous_group.tokens)
#                         block = [previous_group] + block
#                         result = FIM(block)
#                         if result is not None:
#                             final_samples.append(result)
#                         break
#                     else:
#                         block = [group] + block
#                         cur_length += len(group.tokens)
            
#             elif len(blocks) == 0 and len(block) > 0:
#                 result = FIM(block)
#                 if result is not None:
#                     final_samples.append(result)
            
#         else:
#             final_samples.extend(NTP(tokenizer, text, cutoff_len))
    
#     input_ids, attention_mask = [], []
#     for sample in final_samples:
#         input_ids.append(sample["input_ids"] + [end_id])
#         attention_mask.append(sample["attention_mask"] + [1])
    
#     return {"input_ids":input_ids, "attention_mask":attention_mask}

def token_level_FIM_multilines(tokenizer: "PreTrainedTokenizer", text_examples: List[str], cutoff_len: int, overlap_len: int=0) -> Dict[str, List[List[int]]]:
    # 分块时会添加EOS，以及FIM会添加PREFIX、MIDDLE和SUFFIX，所以要预留4个token出来
    cutoff_len -= 4
    
    def split_tokens(res: Dict[str, List[int]], cutoff_len: int, overlap_len) -> List[Dict[str, List[int]]]:
        input_ids_chunks = []
        attention_mask_chunks = []
        
        # 计算每次切片的起始位置
        for i in range(0, len(res["input_ids"]), cutoff_len - overlap_len):
            # 确保切片不会超出列表范围
            if i + cutoff_len <= len(res["input_ids"]):
                input_ids_chunks.append(res["input_ids"][i:i + cutoff_len])
                attention_mask_chunks.append(res["attention_mask"][i:i + cutoff_len])
            else:
                # 处理最后一个切片，确保不会超出范围
                input_ids_chunks.append(res["input_ids"][i:])
                attention_mask_chunks.append(res["attention_mask"][i:])
                break
        
        return [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(input_ids_chunks, attention_mask_chunks)]
    
    def FIM(input_ids:List[int], attention_mask:List[int], pre_id:int, mid_id:int, suf_id:int, end_id:int):
        import random
        
        length = len(input_ids)
        # 10%的概率进行Predit Next Token
        if random.random() > 0.9 or length < 100:
            return (input_ids + [end_id], attention_mask + [1])
        
        
        first, second = random.sample(range(length), 2)
        start, end = min(first, second), max(first, second)
        prefix_ids = input_ids[:start]
        middle_ids = input_ids[start:end]
        suffix_ids = input_ids[end:]
        prefix_mask = attention_mask[:start]
        middle_mask = attention_mask[start:end]
        suffix_mask = attention_mask[end:]
        if random.random() > 0.5:
            # PSM
            output_ids = [pre_id] + prefix_ids + [suf_id] + suffix_ids + [mid_id] + middle_ids + [end_id]
            output_mask = [1] + prefix_mask + [1] + suffix_mask + [1] + middle_mask + [1]
        else:
            # SPM
            output_ids = [pre_id, suf_id] + suffix_ids + [mid_id] + prefix_ids + middle_ids + [end_id]
            output_mask = [1, 1] + suffix_mask + [1] + prefix_mask + middle_mask + [1]
            
        return (output_ids, output_mask)
    
    pre_id = tokenizer.convert_tokens_to_ids("<PREFIX>")
    mid_id = tokenizer.convert_tokens_to_ids("<MIDDLE>")
    suf_id = tokenizer.convert_tokens_to_ids("<SUFFIX>")
    end_id = tokenizer.eos_token_id
    
    split_examples = []
    for text in text_examples:
        res = tokenizer(text, add_special_tokens=False)
        if len(res["input_ids"]) > cutoff_len:
            split_examples.extend(split_tokens(res, cutoff_len, overlap_len))
        else:
            split_examples.append(res)
    
    input_ids, attention_mask = [], []
    for example in split_examples:
        ids, mask = FIM(example["input_ids"], example["attention_mask"], pre_id, mid_id, suf_id, end_id)
        input_ids.append(ids)
        attention_mask.append(mask)
    
    return {"input_ids":input_ids, "attention_mask":attention_mask}

# def process_file(func:Callable[[List[str]], Dict[str, List[List[int]]]], datasets_path:str, output_path:str):
#     # TODO 根据文件类型使用不同的方法
#     table = pd.read_parquet(datasets_path, columns=["text"])
#     samples = []
#     for i, row in table.iterrows():
        
    
# def main():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--datasets_path", help="the path to the raw datasets", required=True)
#     parser.add_argument("--output_path", help="the path where store the processed datasets", required=True)
#     parser.add_argument("--method", choices=["char_level_FIM", "token_level_FIM_multilines"], required=True)
#     parser.add_argument("--cutoff_length", type=int, default=2048)
#     parser.add_argument("--max_workers", type=int, default=4)
    
#     func_list = {
#         "char_level_FIM": char_level_FIM,
#         "token_level_FIM_multilines": token_level_FIM_multilines
#     }
        
#     args = parser.parse_args()
#     datasets_path:str = args.datasets_path
#     output_path:str = args.output_path
#     method:str = args.method
#     cutoff_length:int = args.cutoff_length
#     max_workers:int = args.max_workers
    
#     func = func_list[method]
    
#     print("datasets_path : {}".format(datasets_path))
#     print("otuput_path : {}".format(output_path))
    
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
    
#     files = []
#     if os.path.isdir(datasets_path):
#         for filename in os.listdir(datasets_path):
#             files.append(filename)

#     else:
#         print("unsupport file")
#         exit()
        
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_file, files_group, datasets_path, output_path, is_convert_spaces, i) \
#             for i, files_group in enumerate(files_groups)]
        
#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()

def map_line(code:str) -> List[Tuple[bool, List[int]]]:
    stack = []
    line_map = []
    line_count = 0
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    has_op = False
    has_return = False
    has_import = False
    
    for token_type, string, start, _, _ in tokens:
        # OP
        if token_type == 54:
            has_op = True
            if string == '{' or string == '[' or string == '(':
                stack.append(string)
            elif string == '}' or string == ']' or string == ')':
                stack.pop()
                    
        # NL
        if token_type == 61 and len(stack) == 0:
            line_map.append((False, [start[0] - 1]))
            # print([start[0] - 1])
            line_count = start[0]
        
        # NEWLINE
        # has_op 是为了防止长字符串被当作middle
        # has_return 是因为return不会被当作op
        elif token_type == 4:
            if has_op or has_return or has_import:
                line_map.append((True, [i - 1 for i in range(line_count+1, start[0]+1)]))
            else:
                line_map.append((False, [i - 1 for i in range(line_count+1, start[0]+1)]))
            # print([i - 1 for i in range(line_count+1, start[0]+1)])
            line_count = start[0]
            has_op = False
            has_return = False
            has_import = False
        
        # NAME
        elif token_type == 1:
            if string == "return":
                has_return = True
            elif string == "import":
                has_import = True
    
    return line_map

def char_level_FIM(tokenizer: "PreTrainedTokenizer", text_examples: List[str], cutoff_len: int) -> Dict[str, List[List[int]]]:
    # 需要预留一些token出来
    cutoff_len -= 10

    def split_tokens(res: Dict[str, List[int]], cutoff_len: int) -> List[Dict[str, List[int]]]:
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

    def NTP(code:str) -> List[Dict[str, List[int]]]:
        samples = []
        res = tokenizer(code, add_special_tokens=False)
        if len(res["input_ids"]) > cutoff_len:
            samples.extend(split_tokens(res, cutoff_len))
        else:
            samples.append(res)
        return samples

    class Group:
        def __init__(self, valid:bool = False, code_snippet:str = "", \
            tokens:Optional[List[int]] = None, mask:Optional[List[int]] = None) -> None:
            self.valid:bool = valid
            self.code_snippet:str = code_snippet
            self.tokens:Optional[List[int]] = tokens
            self.mask:Optional[List[int]] = mask

    def FIM(block:List[Group]) -> Optional[Dict[str, List[int]]]:
        valid_group_ids = [i for i, group in enumerate(block) if group.valid]
        if len(valid_group_ids) != 0:
            if len(valid_group_ids) == 1:
                selected_group_id = valid_group_ids[0]
            else:
                selected_group_id = valid_group_ids[random.randint(0, len(valid_group_ids) -1)]
            selected_group = block[selected_group_id]
            position = random.randint(0, len(selected_group.code_snippet))
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
            
            res = tokenizer(code, add_special_tokens=False) 
                    
            return {"input_ids": res["input_ids"], "attention_mask": res["attention_mask"]}

        else:
            return None


    end_id = tokenizer.eos_token_id

    final_samples:List[Dict[str, List[int]]] = []
    for text in text_examples:
        if True:
            if random.random() > 0.9:
                final_samples.extend(NTP(text))
                continue
                
            line_map = map_line(text.replace("<UNK>", "zxcv"))
            raw_lines = text.split("\n")
            raw_lines = [line+"\n" if i != len(raw_lines) - 1 else line for i, line in enumerate(raw_lines)]
            line_group:List[Group] = []
            for valid, line_idx in line_map:
                # print("".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx]),end="")
                if len(line_group) == 0:
                    line_group.append(Group(valid, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
                else:
                    if not valid and valid == line_group[-1].valid:
                        line_group[-1].code_snippet += "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])
                    else:
                        line_group.append(Group(valid, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
            
            blocks:List[List[Group]] = []
            block:List[Group] = []
            cur_length = 0
            for group in line_group:
                res = tokenizer(group.code_snippet, add_special_tokens=False)
                
                # print(group.code_snippet, end="")
                # print("group length " + str(len(res["input_ids"])) + " current length " + str(cur_length))
                
                # 一个语句就能塞满整个上下文窗口的话，这种数据不要也罢
                if len(res["input_ids"]) > cutoff_len:
                    continue
                
                group.tokens = res["input_ids"]
                group.mask = res["attention_mask"]
                # print(cur_length, len(group.tokens))
                if cur_length + len(group.tokens) > cutoff_len:
                    # print(cur_length, len(group.tokens))
                    next_group = Group(False, "",
                        res["input_ids"][cutoff_len-cur_length-len(group.tokens):],
                        res["attention_mask"][cutoff_len-cur_length-len(group.mask):])
                    next_group.code_snippet = tokenizer.decode(next_group.tokens)
                    group.valid = False
                    group.tokens = res["input_ids"][:cutoff_len-cur_length-len(group.tokens)]
                    group.mask = res["attention_mask"][:cutoff_len-cur_length-len(group.mask)]
                    group.code_snippet = group.code_snippet[:-len(next_group.code_snippet)]
                    block.append(group)
                    # print(cur_length, len(group.tokens))
                    # print(group.code_snippet)
                    # print(next_group.code_snippet)
                    result = FIM(block)
                    if result is not None:
                        final_samples.append(result)
                    
                    blocks.append(block)
                    cur_length = len(next_group.tokens)
                    block = [next_group]
                    
                elif cur_length + len(group.tokens) == cutoff_len:
                    block.append(group)
                    result = FIM(block)
                    if result is not None:
                        final_samples.append(result)
                    
                    blocks.append(block)
                    cur_length = 0
                    block = []
                
                else:
                    block.append(group)
                    cur_length += len(group.tokens)
            
            # print(len(block))
            # for group in block:
            #     print(">" + group.code_snippet)
                    
            if len(blocks) != 0 and len(block) > 0:
                for group in reversed(blocks[-1]):
                    if cur_length + len(group.tokens) >= cutoff_len:
                        previous_group = Group(False, "", 
                            group.tokens[cur_length-cutoff_len:],
                            group.mask[cur_length-cutoff_len:])
                        previous_group.code_snippet = tokenizer.decode(previous_group.tokens)
                        # print(previous_group.code_snippet)
                        block = [previous_group] + block
                        result = FIM(block)
                        if result is not None:
                            final_samples.append(result)
                        break
                    else:
                        block = [group] + block
                        cur_length += len(group.tokens)
            
            elif len(blocks) == 0 and len(block) > 0:
                result = FIM(block)
                if result is not None:
                    final_samples.append(result)
            
        else:
            final_samples.extend(NTP(text))

    input_ids, attention_mask = [], []
    for sample in final_samples:
        input_ids.append(sample["input_ids"] + [end_id])
        attention_mask.append(sample["attention_mask"] + [1])

    return {"input_ids":input_ids, "attention_mask":attention_mask}