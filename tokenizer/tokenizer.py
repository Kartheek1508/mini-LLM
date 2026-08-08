from tokenizers import ByteLevelBPETokenizer
import os
base_dir = os.path.dirname(os.path.abspath(__file__))

vocab_path = os.path.join(base_dir, "trained_tokenizer", "vocab.json")
merges_path = os.path.join(base_dir, "trained_tokenizer", "merges.txt")

tokenizer = ByteLevelBPETokenizer(
    vocab_path,
    merges_path)

def tokenize(text):
    encoded = tokenizer.encode(text)
    return encoded.ids

def decode(tokens):
    return tokenizer.decode(tokens)