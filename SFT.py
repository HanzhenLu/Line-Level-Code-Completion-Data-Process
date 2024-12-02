import datasets
from transformers import AutoTokenizer

def tokenize_batch(batch, tokenizer):
    return_batch = {"input_ids":[], "attention_mask":[]}
    for code in batch["text"]:
        
        # res = split_and_tokenize(tokenizer, code, ["<EOS>", "<BOS>", "<UNK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<PAD>", "<MASK>"])
        return_batch["input_ids"].append(res["input_ids"])
        return_batch["attention_mask"].append(res["attention_mask"])
    return return_batch

tokenizer = AutoTokenizer.from_pretrained("tokenizer_bbpe_keywords")
datasets.load_from_disk("pretrained-high-quality")

