import os
import keyword
from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from transformers import LlamaTokenizerFast

datasets_path = "/data/hanzhenlu/dataset/Stack-V2-python-spaces-txt"
save_path = "tokenizer_bbpe_keywords"

if not os.path.exists(save_path):
    os.makedirs(save_path)

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

# tokenizer.add_special_tokens(["<UNK>", "<BOS>", "<EOS>", "<PAD>", "<MASK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<4BLANK>", "<8BLANK>", "<16BLANK>"])
tokenizer.add_special_tokens(["<UNK>", "<BOS>", "<EOS>", "<PAD>", "<MASK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>"])
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=16000, show_progress = True, initial_alphabet = keyword.kwlist, \
    # special_tokens=["<UNK>", "<BOS>", "<EOS>", "<PAD>", "<MASK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>", "<4BLANK>", "<8BLANK>", "<16BLANK>"])
    special_tokens=["<UNK>", "<BOS>", "<EOS>", "<PAD>", "<MASK>", "<PREFIX>", "<SUFFIX>", "<MIDDLE>"])
files = [os.path.join(datasets_path, file) for file in os.listdir(datasets_path)]
# files = ["/data/hanzhenlu/dataset/python_txt/0.txt"]

tokenizer.train(files=files, trainer=trainer)

tokenizer_model_path = os.path.join(save_path, "tokenizer.json")
tokenizer.save(tokenizer_model_path)

tokenizer = LlamaTokenizerFast(tokenizer_file=tokenizer_model_path, bos_token="<BOS>", eos_token="<EOS>", unk_token="<UNK>")
tokenizer.save_pretrained(save_path)