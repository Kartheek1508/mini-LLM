from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

target = 2 * 1024**3  # 2 GB

written = 0

with open("fineweb_2gb.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"] + "\n"

        f.write(text)

        written += len(text.encode("utf-8"))

        if written >= target:
            break

print(f"Wrote {written / 1024**3:.2f} GB")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files = ["fineweb_2gb.txt"],
               vocab_size = 35000,
               min_frequency = 2)

tokenizer.save_model("trained_tokenizer")
