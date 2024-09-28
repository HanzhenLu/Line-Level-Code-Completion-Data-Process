import ast
import string
import tokenize
import io
from typing import Optional

def check_syntax(code:str, blob_id:str) -> bool:
    try:
        _ = ast.parse(code, filename=blob_id)
        return True
    
    except:
        return False
    
def check_todo(code:str) -> bool:
    total_length = len(code)
    
    try:
        # 使用 tokenize 模块来识别注释
        comments = []
        strings_length = 0
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token_type, token_string, _, _, _ in tokens:
            if token_type == tokenize.COMMENT:
                comments.append(token_string)
            elif token_type == tokenize.STRING:
                strings_length += len(token_string)
        
        # 检查注释中是否包含 TODO
        for comment in comments:
            if "TODO" in comment:
                return False
        
        comments_length = sum([len(comment) for comment in comments])
        if (comments_length + strings_length) > 0.5 * total_length:
            return False
        
        return True
    except:
        return False

def check_empty_function(code:str) -> bool:
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_body = node.body
                
                if len(function_body) == 1:
                    if isinstance(function_body[0], ast.Pass):
                        print(code)
                        return False
        
        return True
    except:
        print(code)
        return False
    
def check_code_length(code:str) -> bool:
    
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    if not non_empty_lines:
        return False

    max_length = max(len(line) for line in non_empty_lines)
    average_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)

    if max_length > 1000 or average_length > 100:
        return False
    else:
        return True
    
def check_and_update_characters(code:str, ratio:float=0.9) -> Optional[str]:
    allowed_characters = set(string.printable)
    allowed_characters.update(chr(i) for i in range(0x4e00, 0x9fff + 1))
    # 添加中文标点符号
    chinese_punctuation = "，。？！：；“”‘’（）【】《》——…"
    allowed_characters.update(chinese_punctuation)

    # 记录非法字符的位置
    invalid_idx = [idx for idx, char in enumerate(code) if char not in allowed_characters]
    
    invalid_count = len(invalid_idx)
    total_count = len(code)

    if total_count != 0 and (total_count - invalid_count) / total_count > ratio:
        return code
    else:
        # 将所有非法字符替换为<UNK>
        for idx in reversed(invalid_idx):
            code = code[:idx] + "<UNK>" + code[idx+1:]
        return None