from pre_processing import preprocess
from tokenizer.tokenizer import tokenize
from tokenizer.tokenizer import decode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os

from datasets import load_dataset

class Token_dataset(Dataset):
        def __init__(self,data):
            self.data = data

        def __getitem__(self, index):
            start = np.random.randint(0, len(self.data) - 1024)
            chunk = self.data[start:start + 1024].copy()
            return torch.tensor(chunk, dtype=torch.long)

        def __len__(self):
            return len(self.data)

def main():
#loading dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    filename = "train.bin"#file for the memmap

    with open(filename,"wb") as f:
        batch = []
        batch_size = 32#doing in batches cause doing it all at once is inefficent
        for sample in dataset.take(1000):
            batch.append(sample["text"])

            if len(batch) == batch_size:
                docs = preprocess(batch)
                token_ids = []
                for doc in docs:
                    token_ids.extend(tokenize(doc))
                arr = np.asarray(token_ids,dtype=np.uint16) #directly storing it to a binary file instead of using np.memmap()
                arr.tofile(f)
                batch = []

        if batch: #this entire if is just when there aren't batch_size docs left
                docs = preprocess(batch)
                token_ids = []
                for doc in docs:
                    token_ids.extend(tokenize(doc))
                arr = np.asarray(token_ids,dtype=np.uint16)
                arr.tofile(f)

    tokens = np.memmap( #just accesing the tokens we made above
        "train.bin",
        dtype=np.uint16,
        mode="r"
    )

    print("Total tokens:", len(tokens))
    print("File size:", os.path.getsize("train.bin"), "bytes")


    data = np.memmap("train.bin",dtype=np.uint16,mode="r")


    data = np.memmap("train.bin",dtype=np.uint16,mode="r")

    dataset = Token_dataset(data)

    loader = DataLoader(dataset,batch_size=8)

    batch = next(iter(loader))
    print(batch.shape)#should be (8,1024) cause __getitem__ only gives 1024 and 8 cause we set batch_size=8 


    sample = batch[0].tolist()

    text = decode(sample)
    print(text)


if __name__ =="__main__":
    main()