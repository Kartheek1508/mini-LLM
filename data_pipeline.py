from pre_processing import preprocess
from tokenizer.tokenizer import tokenize
from tokenizer.tokenizer import decode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import json
from datasets import load_dataset, DownloadConfig

download_config = DownloadConfig(max_retries=10, token=True)

class Token_dataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        start = np.random.randint(0,len(self.data) - self.seq_len)

        chunk = self.data[start:start + self.seq_len].copy()

        return torch.tensor(chunk, dtype=torch.long)

    def __len__(self):
        return len(self.data)

def main():
    TARGET_TOKENS = 1_000_000

    dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
    download_config=download_config
)

    filename = "train.bin"
    progress_file = "progress.json"

    batch_size = 32
    last_reported = 0

    # Load previous progress
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)

        total_tokens = progress["tokens"]
        documents_processed = progress["documents"]
        checkpoint_bytes = progress["file_bytes"]

        print(
            f"Resuming from {total_tokens:,} tokens "
            f"and {documents_processed:,} documents"
        )

        # Remove anything after the last confirmed checkpoint
        with open(filename, "r+b") as f:
            f.truncate(checkpoint_bytes)

        # Skip documents that were already processed
        dataset = dataset.skip(documents_processed)

    else:
        total_tokens = 0
        documents_processed = 0
        checkpoint_bytes = 0

        print("Starting from the beginning")

    last_reported = total_tokens

    batch = []

    with open(filename, "ab") as f:

        for sample in dataset:

            batch.append(sample["text"])

            if len(batch) == batch_size:

                docs = preprocess(batch)

                token_ids = []

                for doc in docs:
                    token_ids.extend(tokenize(doc))

                remaining = TARGET_TOKENS - total_tokens

                if len(token_ids) > remaining:
                    token_ids = token_ids[:remaining]

                arr = np.asarray(token_ids, dtype=np.uint16)
                arr.tofile(f)

                total_tokens += len(token_ids)
                documents_processed += len(batch)

                batch = []

                # Make sure the data is written before checkpointing
                f.flush()
                os.fsync(f.fileno())

                checkpoint_bytes = f.tell()

                # Save checkpoint
                temp_file = progress_file + ".tmp"

                with open(temp_file, "w") as pf:
                    json.dump({
                        "tokens": total_tokens,
                        "documents": documents_processed,
                        "file_bytes": checkpoint_bytes
                    }, pf)

                    pf.flush()
                    os.fsync(pf.fileno())

                os.replace(temp_file, progress_file)

                # Print every ~10M tokens
                if total_tokens - last_reported >= 10_000_000:
                    print(f"Tokens: {total_tokens:,} / {TARGET_TOKENS:,}")
                    last_reported = total_tokens

                if total_tokens >= TARGET_TOKENS:
                    break

        # Process remaining documents
        if batch and total_tokens < TARGET_TOKENS:

            docs = preprocess(batch)

            token_ids = []

            for doc in docs:
                token_ids.extend(tokenize(doc))

            remaining = TARGET_TOKENS - total_tokens
            token_ids = token_ids[:remaining]

            arr = np.asarray(token_ids, dtype=np.uint16)
            arr.tofile(f)

            total_tokens += len(token_ids)
            documents_processed += len(batch)

            f.flush()
            os.fsync(f.fileno())

            checkpoint_bytes = f.tell()

            temp_file = progress_file + ".tmp"

            with open(temp_file, "w") as pf:
                json.dump({
                    "tokens": total_tokens,
                    "documents": documents_processed,
                    "file_bytes": checkpoint_bytes
                }, pf)

                pf.flush()
                os.fsync(pf.fileno())

            os.replace(temp_file, progress_file)

    print(f"Total tokens: {total_tokens:,}")
    print("File size:", os.path.getsize(filename), "bytes")


if __name__ == "__main__":
    main()
