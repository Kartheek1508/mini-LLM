
from collections import Counter
from langdetect import detect
from datasketch import MinHash, MinHashLSH
from tokenizer.tokenizer import tokenize
import numpy as np
import os


def preprocess(docs):

    def get_word_ngrams(text, n=5):

        words = text.lower().split()

        ngrams = []

        for i in range(len(words) - n + 1):

            piece = ""

            for j in range(n):
                if j == 0:
                    piece = words[i + j]
                else:
                    piece = piece + " " + words[i + j]

            ngrams.append(piece)

        return ngrams


    def deduplicate(docs, similarity_threshold=0.75, num_hashes=128):

        lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_hashes)

        kept = []

        for i in range(len(docs)):

            doc = docs[i]

            if len(doc.split()) < 10:
                kept.append(doc)

            else:

                signature = MinHash(num_perm=num_hashes)

                ngrams = get_word_ngrams(doc)

                for item in ngrams:
                    signature.update(item.encode("utf-8"))

                result = lsh.query(signature)

                if len(result) == 0:
                    lsh.insert("doc_" + str(i), signature)
                    kept.append(doc)

        return kept


    # language filter
    kept = []

    for doc in docs:
        if len(doc) < 50:
            kept.append(doc)
        else:
            try:
                language = detect(doc)
                if language == "en":
                    kept.append(doc)
            except:
                pass

    docs = kept


    # adult content filter
    adult_words = [
        "porn",
        "xxx",
        "nude",
        "naked",
        "escort",
        "adult content",
        "explicit",
        "nsfw"
    ]

    kept = []

    for doc in docs:

        found = False

        for word in adult_words:
            if word in doc.lower():
                found = True

        if found == False:
            kept.append(doc)

    docs = kept


    # deduplication
    docs = deduplicate(docs)


    # c4 filters
    cookie_phrases = [
        "cookie policy",
        "use of cookies",
        "uses cookies",
        "accept cookies",
        "cookie notice"
    ]

    kept = []

    for doc in docs:
        if "javascript" not in doc.lower():
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        found = False

        for phrase in cookie_phrases:
            if phrase in doc.lower():
                found = True

        if found == False:
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:
        if "lorem ipsum" not in doc.lower():
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:
        if "{" not in doc:
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        word_count = len(doc.split())

        if word_count >= 50 and word_count <= 100000:
            kept.append(doc)

    docs = kept


    # statistical filters
    sentence_endings = set(".!?\"'")

    kept = []

    for doc in docs:

        lines = []

        for line in doc.split("\n"):
            if line.strip() != "":
                lines.append(line.strip())

        if len(lines) > 0:

            count = 0

            for line in lines:
                if line[-1] in sentence_endings:
                    count = count + 1

            ratio = count / len(lines)

            if ratio >= 0.12:
                kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        lines = []

        for line in doc.split("\n"):
            if line.strip() != "":
                lines.append(line.strip())

        if len(lines) > 0:

            total_chars = 0

            for line in lines:
                total_chars = total_chars + len(line)

            line_counts = Counter(lines)

            duplicate_chars = 0

            for line in line_counts:

                count = line_counts[line]

                if count > 1:
                    duplicate_chars = duplicate_chars + (len(line) * (count - 1))

            ratio = duplicate_chars / total_chars

            if ratio <= 0.10:
                kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        lines = []

        for line in doc.split("\n"):
            if line.strip() != "":
                lines.append(line.strip())

        if len(lines) > 0:

            short_lines = 0

            for line in lines:
                if len(line) < 30:
                    short_lines = short_lines + 1

            ratio = short_lines / len(lines)

            if ratio <= 0.67:
                kept.append(doc)

    docs = kept

    tokenized = tokenize(docs)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    return docs


'''
This can be used for any dataset by using the following method
from datasets import load_dataset
# load dataset
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

batch = []

for sample in dataset.take(1000):
    batch.append(sample["text"])
docs = preprocessed(batch)

print(f"{len(docs)} clean documents")
print(f" removed : {len(batch) - len(docs)} documents")'''