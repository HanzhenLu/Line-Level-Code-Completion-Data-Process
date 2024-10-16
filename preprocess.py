import tokenize
import io
import random
import re
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional

def tokenize_and_split(tokenizer: "PreTrainedTokenizer", text_examples: List[str], cutoff_len: int) -> Dict[str, List[List[int]]]:
    # 需要预留一些token出来
    cutoff_len -= 10
    
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
    
    def NTP(code:str) -> List[Dict[str, List[int]]]:
        samples = []
        res = split_and_tokenize(tokenizer, code, ["<EOS>", "<BOS>", "<UNK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<PAD>", "<MASK>"])
        if len(res["input_ids"]) > cutoff_len:
            samples.extend(segment(res, cutoff_len))
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
    
    def char_level_FIM(block:List[Group]) -> Optional[Dict[str, List[int]]]:
        valid_group_ids = [i for i, group in enumerate(block) if not group.code_snippet.isspace() and group.code_snippet != "" and len(group.code_snippet) < 200]
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
                    
            return {"input_ids": res["input_ids"], "attention_mask": res["attention_mask"]}

        else:
            return None
        
    def token_leval_FIM(block:List[Group]) -> Optional[Dict[str, List[int]]]:
        valid_group_ids = [i for i, group in enumerate(block) if group.valid]
        if len(valid_group_ids) != 0:
            if len(valid_group_ids) == 1:
                selected_group_id = valid_group_ids[0]
            else:
                selected_group_id = valid_group_ids[random.randint(0, len(valid_group_ids) -1)]
            selected_group = block[selected_group_id]
            position = random.randint(0, len(selected_group.tokens) - 2)
            middle_tokens = selected_group.tokens[position:]
            middle_code = tokenizer.decode(middle_tokens)
            prefix_code = selected_group.code_snippet[:-len(middle_code)]
            
            prefix_code = "".join(group.code_snippet for group in block[:selected_group_id]) + prefix_code
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
    
    def map_line(code:str) -> List[List[int]]:
        stack = []
        line_map = []
        line_count = 0
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        
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
            # has_op 是为了防止长字符串被当作middle
            # has_keywords和has_op 是为了防止长字符串被当作middle
            elif token_type == 4:
                line_map.append([i - 1 for i in range(line_count+1, start[0]+1)])
                line_count = start[0]
        
        return line_map
    
    end_id = tokenizer.eos_token_id
    # BPE dropoout
    # tokenizer._tokenizer.model.dropout = 0.1
    
    final_samples:List[Dict[str, List[int]]] = []
    for text in text_examples:
        if True:
            
            if True:
                FIM = char_level_FIM
            elif data_args.FIM_method == "token-level":
                FIM = token_leval_FIM
            else:
                raise ValueError("`FIM_method` only supports char-level or token-level.")
            
            if random.random() > 0.9:
                final_samples.extend(NTP(text))
                continue
                
            line_map = map_line(text.replace("<UNK>", "zxcv"))
            raw_lines = text.split("\n")
            raw_lines = [line+"\n" if i != len(raw_lines) - 1 else line for i, line in enumerate(raw_lines)]
            line_group:List[Group] = []
            for line_idx in line_map:
                valid = True
                if len(line_group) == 0:
                    line_group.append(Group(valid, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
                else:
                    line_group.append(Group(valid, "".join([raw_lines[i].replace("zxcv", "<UNK>") for i in line_idx])))
            
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
                    next_group = Group(False, "",
                        res["input_ids"][cutoff_len-cur_length-len(group.tokens):],
                        res["attention_mask"][cutoff_len-cur_length-len(group.mask):])
                    next_group.code_snippet = tokenizer.decode(next_group.tokens)
                    group.valid = False
                    group.tokens = res["input_ids"][:cutoff_len-cur_length-len(group.tokens)]
                    group.mask = res["attention_mask"][:cutoff_len-cur_length-len(group.mask)]
                    group.code_snippet = group.code_snippet[:-len(next_group.code_snippet)]
                    block.append(group)
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
                    
            if len(blocks) != 0 and len(block) > 0:
                for group in reversed(blocks[-1]):
                    if cur_length + len(group.tokens) >= cutoff_len:
                        previous_group = Group(False, "", 
                            group.tokens[cur_length-cutoff_len:],
                            group.mask[cur_length-cutoff_len:])
                        previous_group.code_snippet = tokenizer.decode(previous_group.tokens)
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