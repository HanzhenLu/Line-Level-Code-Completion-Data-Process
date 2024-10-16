import json
from typing import Dict
from transformers import AutoTokenizer

'''
统计词表中有多少个中英文以外的词
需要指定
file: 分词器的路径
'''

file = "/nasdata/Model/CodeLlama-7b-hf"


def classify_char(char: str) -> bool:
    if '\u4e00' <= char <= '\u9fff':
        return True
    elif char.isalpha():
        return True
    elif char.isascii() and char.isprintable() and not char.isalnum():
        return True
    else:
        return False

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(file)

    vocab = tokenizer.vocab

    count = 0
    for key in vocab.keys():
        count += 1
        for i in range(len(key)):
            if not classify_char(key[i]):
                count -= 1
                break
            
    print(count)