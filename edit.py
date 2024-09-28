import re
import sys
import os
from typing import List, Tuple, Optional

# 提取文件开头的注释内容
def extract_leading_comments(lines:List[str]) -> Tuple[List[str], Optional[int]]:
    comments = []
    in_multiline_comment = False
    valid_content_begin = None

    for idx, line in enumerate(lines):
        # 去除行首和行尾的空白字符
        line = line.strip()
        
        # 检查是否在多行注释中 '''
        if in_multiline_comment == "single":
            if re.search(r"[\']{3}[\']{3}", line):
                partition = line.index("''''''")
                comments[-1] += '\n' + line[:partition+3]
                comments.append(line[partition+3:])
                continue
            elif re.search(r"[\']{3}[\"]{3}", line):
                in_multiline_comment = "double"
                partition = line.index("'''\"\"\"")
                comments[-1] += '\n' + line[:partition+3]
                comments.append(line[partition+3:])
                continue
            elif re.search(r"[\']{3}", line):
                in_multiline_comment = False
            comments[-1] += '\n' + line
            continue
        
        elif in_multiline_comment == "double":
            if re.search(r"[\"]{3}[\']{3}", line):
                in_multiline_comment = "single"
                partition = line.index("\"\"\"'''")
                comments[-1] += '\n' + line[:partition+3]
                comments.append(line[partition+3:])
                continue
            elif re.search(r"[\"]{3}[\"]{3}", line):
                partition = line.index('""""""')
                comments[-1] += '\n' + line[:partition+3]
                comments.append(line[partition+3:])
                continue
            elif re.search(r"[\"]{3}", line):
                in_multiline_comment = False
            comments[-1] += '\n' + line
            continue
        

        # 检查单行注释
        if line.startswith('#'):
            comments.append(line)
        # 检查多行注释开始 '''
        elif re.match(r"\s*[\']{3}", line):
            comments.append(line)
            if not re.search(r"[\']{3}.*[\']{3}.*$", line):
                in_multiline_comment = "single"
        
        # 检查多行注释开始 """
        elif re.match(r"\s*[\"]{3}", line):
            comments.append(line)
            if not re.search(r"[\"]{3}.*[\"]{3}.*$", line):
                in_multiline_comment = "double"
        
        elif line == "":
            comments.append(line)
            continue
        
        # 如果遇到非注释内容，停止处理
        else:
            valid_content_begin = idx
            break

    return comments, valid_content_begin

# 根据一些关键词来推测注释的类型
def detect_keywords(comment: str) -> str:
    shebang_format = re.compile(r"#\s*!.*python.*")
    coding_format = re.compile(r".*(coding=|-\*-\s*coding:).*")
    copyright_format = re.compile( \
        r".*(Copyright \(C\)|All rights reserved|THE SOFTWARE IS PROVIDED|Apache License|MIT LICENSE|MIT LICENCE|BSD.*License|License.*BSD|SPDX-License-Identifier|GNU General Public License).*", \
        re.IGNORECASE | re.DOTALL)
    pylint_format = re.compile(r"#\s*pylint.*")
    
    if re.match(shebang_format, comment):
        return "shebang"
    
    elif re.match(coding_format, comment):
        return "coding"

    elif re.match(copyright_format, comment):
        return "copyright"
    
    elif re.match(pylint_format, comment):
        return "pylint"
    
    else:
        return ""

'''只使用stack没法处理多行参数、多行注释这些特殊情况
def convert_spaces(lines:List[str]) -> Optional[List[str]]:
    stack = [0]
    
    for i in range(len(lines)):
        if lines[i].isspace():
            lines[i] = ""
            continue
        
        if lines[i] == "" or lines[i].strip().startswith('#'):
            continue
        
        match = re.match(r"^( *)\t", lines[i])
        while match:
            leading_spaces = match.group(1)
            spaces_num = len(leading_spaces)
            step_in = spaces_num // 4 + 1
            lines[i] = re.sub(r"^( *)\t", " " * 4 * step_in, lines[i])
            match = re.match(r"^( *)\t", lines[i])
        
        match = re.match(r"^( +)", lines[i])
        if match:
            leading_spaces = match.group(1)
            spaces_num = len(leading_spaces)
        else:
            spaces_num = 0
        
        

        former_indent = stack[-1]
        current_indent = spaces_num
        
        if current_indent > former_indent:
            lines[i] = "<STEPIN>" + lines[i].strip()
            stack.append(current_indent)
            
        elif current_indent < former_indent:
            num = 0
            try:
                while stack[-1] != current_indent:
                    num += 1
                    stack.pop()
            except:
                return None
                
            lines[i] = "<STEPOUT>" * num + lines[i].strip()
        else:
            lines[i] = lines[i].strip()
        
        former_indent = current_indent
    
    return lines

def reset_spaces(code:str) -> str:
    current_indent = 0
    lines = code.split('\n')
    for i in range(len(lines)):
        if lines[i].isspace():
            lines[i] = ""
            continue
        if lines[i].startswith("<STEPIN>"):
            num = lines[i].count("<STEPIN>")
            current_indent += num
            lines[i] = ' ' * 4 * current_indent + lines[i][8*num:]
        elif lines[i].startswith("<STEPOUT>"):
            num = lines[i].count("STEPOUT")
            current_indent -= num
            lines[i] = ' ' * 4 * current_indent + lines[i][9*num:]
        else:
            lines[i] = ' ' * 4 * current_indent + lines[i]
    
    return '\n'.join(lines)'''

def convert_spaces(lines:List[str]) -> List[str]:
    
    for i in range(len(lines)):
        if lines[i].isspace():
            lines[i] = ""
            continue
        
        if lines[i] == "" or lines[i].strip().startswith('#'):
            continue
        
        # lines[i] = lines[i].replace(' '*16, "<16BLANK>")
        # lines[i] = lines[i].replace(' '*8, "<8BLANK>")
        lines[i] = lines[i].replace(' '*4, "<4BLANK>")
        
    return lines

def reset_spaces(code:str) -> str:
    lines = code.split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].replace("<4BLANK>", ' '*4)
        lines[i] = lines[i].replace("<8BLANK>", ' '*8)
        lines[i] = lines[i].replace("<16BLANK>", ' '*16)
        
    
    return '\n'.join(lines)

            

def edit_string(input_string: str, is_convert_spaces: bool) -> Optional[str]:
    context = input_string.split('\n')
        
    comments, idx = extract_leading_comments(context)
    if idx is None:
        return None
    
    # 使用数组记录每一行注释是否保留
    # -1 未访问
    # 0 不保留
    # 1 已访问但暂未决定
    # 2 保留
    valid = [-1] * len(comments)
    
    # 如果copyright是以#的格式存在，需要删除该行上下文中所有以#开头的注释    
    is_copyright = False
    
    for i, comment in enumerate(comments):
        
        if is_copyright and comment.startswith('#'):
            valid[i] = 0
            continue
        else:
            is_copyright = False
            
        
        result = detect_keywords(comment)
        if result == "shebang" or result == "coding" or result == "pylint":
            valid[i] = 0
        elif result == "copyright":
            valid[i] = 0
            # 如果以#开头，copyright可能涉及多个单行注释
            if comment.startswith('#'):
                valid = [0 if i == 1 else i for i in valid]
                is_copyright = True
        elif comment.startswith('#'):
            valid[i] = 1
        else:
            valid[i] = 2
            valid = [2 if i == 1 else i for i in valid]
                
    valid_comment = [comment for i, comment in zip(valid, comments) if i == 2 or i == 1]
    valid_comment = [comment.strip() + '\n' for comment in valid_comment]
    
    if is_convert_spaces:
        code = convert_spaces(context[idx:])
    else:
        code = context[idx:]
    
    new_context = valid_comment + [line + '\n' for line in code]
    
    return ''.join(new_context)

