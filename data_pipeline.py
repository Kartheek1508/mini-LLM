from pre_processing import preprocess
from tokenizer.tokenizer import tokenize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
import numpy as np
import torch
import os
import json

class Token_dataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        start = np.random.randint(0, len(self.data) - self.seq_len)
        chunk = self.data[start:start + self.seq_len].copy()
        return torch.tensor(chunk, dtype=torch.long)

    def __len__(self):
        return len(self.data)

def save_checkpoint(progress_file, total_tokens, shard_index, rows_processed, file_bytes):
    temp_file = progress_file + ".tmp"

    with open(temp_file, "w") as f:
        json.dump({"tokens": total_tokens, "shard": shard_index, "rows": rows_processed, "file_bytes": file_bytes}, f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_file, progress_file)

def main():
    TARGET_TOKENS = 2_000_000

    repo_id = "HuggingFaceFW/fineweb-edu"
    revision = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

    filename = "train.bin"
    progress_file = "progress.json"
    temp_dir = "parquet_temp"

    batch_size = 32
    last_reported = 0

    os.makedirs(temp_dir, exist_ok=True)

    api = HfApi()

    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)

    shards = sorted([file for file in files if file.startswith("sample/10BT/") and file.endswith(".parquet")])

    print(f"Found {len(shards)} Parquet shards")

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)

        total_tokens = progress["tokens"]
        shard_index = progress["shard"]
        rows_processed = progress["rows"]
        checkpoint_bytes = progress["file_bytes"]

        print(f"Resuming from {total_tokens:,} tokens, shard {shard_index}, {rows_processed:,} rows")

        with open(filename, "r+b") as f:
            f.truncate(checkpoint_bytes)
    else:
        total_tokens = 0
        shard_index = 0
        rows_processed = 0
        checkpoint_bytes = 0

        print("Starting from the beginning")

    with open(filename, "ab") as output:
        for current_shard in range(shard_index, len(shards)):
            shard = shards[current_shard]

            print(f"Processing shard {current_shard + 1}/{len(shards)}")
            print(shard)

            local_path = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset", revision=revision, local_dir=temp_dir, token=True)

            parquet_file = pq.ParquetFile(local_path)

            rows_to_skip = rows_processed if current_shard == shard_index else 0
            rows_seen = 0

            for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"], use_threads=False):
                if rows_seen + len(record_batch) <= rows_to_skip:
                    rows_seen += len(record_batch)
                    continue

                texts = record_batch["text"].to_pylist()

                if rows_to_skip > rows_seen:
                    offset = rows_to_skip - rows_seen
                    texts = texts[offset:]

                rows_seen += len(record_batch)

                docs = preprocess(texts)

                token_ids = []

                for doc in docs:
                    token_ids.extend(tokenize(doc))

                remaining = TARGET_TOKENS - total_tokens

                if len(token_ids) > remaining:
                    token_ids = token_ids[:remaining]

                arr = np.asarray(token_ids, dtype=np.uint16)
                arr.tofile(output)

                total_tokens += len(token_ids)
                rows_processed = rows_seen

                output.flush()
                os.fsync(output.fileno())

                checkpoint_bytes = output.tell()

                save_checkpoint(progress_file, total_tokens, current_shard, rows_processed, checkpoint_bytes)

                if total_tokens - last_reported >= 10_000_000:
                    print(f"Tokens: {total_tokens:,} / {TARGET_TOKENS:,}")
                    last_reported = total_tokens

                if total_tokens >= TARGET_TOKENS:
                    break

            parquet_file.close()
            os.remove(local_path)

            rows_processed = 0

            if total_tokens >= TARGET_TOKENS:
                break

            shard_index = current_shard + 1

            save_checkpoint(progress_file, total_tokens, shard_index, 0, output.tell())

    print(f"Total tokens: {total_tokens:,}")
    print("File size:", os.path.getsize(filename), "bytes")

if __name__ == "__main__":
    main()