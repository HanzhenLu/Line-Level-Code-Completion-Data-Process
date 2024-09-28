import json

file = "/nasdata/Model/CodeLlama-7b-hf/tokenizer.json"
# file = "/nasdata/Model/deepseek-coder-6.7b-base/tokenizer.json"
with open(file, 'r') as f:
    js = json.loads(f.read())
    
model = js["model"]
vocab = model["vocab"]

def classify_char(char: str) -> bool:
    if '\u4e00' <= char <= '\u9fff':
        return True
    elif char.isalpha():
        return True
    elif char.isascii() and char.isprintable() and not char.isalnum():
        return True
    else:
        return False

count = 0
for key in vocab.keys():
    count += 1
    for i in range(len(key)):
        if not classify_char(key[i]):
            count -= 1
            break
        
print(count)