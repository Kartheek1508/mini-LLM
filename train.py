import wandb
from data_pipeline import Token_dataset
from torch.utils.data import DataLoader
from model.transformer_block import Transformer,causal_mask
from torch.optim import AdamW,lr_scheduler
from torch.nn import CrossEntropyLoss
import numpy as np
import time
import torch

vocab_size = 35000
d_model = 512
num_layers = 4
heads = 8
intermediate_dim = 2048
seq_len = 512
batch_size = 2
use_amp = True
accumilation_steps = 2
model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    heads=heads,
    intermediate_dim=intermediate_dim
)


optimizer = AdamW(model.parameters(),lr=3e-4,betas=(0.9, 0.999),eps=1e-8,weight_decay=0.01)

warmup_steps =5
total_steps = 20
min_lr = 1e-5
max_lr = 3e-4

def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def lr_lambda(step):
    # Warmup
    if step < warmup_steps:
        return step / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))

    return (min_lr / max_lr) + (1 - min_lr / max_lr) * cosine

scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda
)

data = np.memmap("train.bin",dtype=np.uint16,mode="r")

dataset = Token_dataset(data)

loader = DataLoader(
    dataset,
    batch_size=batch_size
)

wandb.init(
    project="mini-llm",
    entity="bnsk",
    config={
        "precision": "bf16" if use_amp else "fp32",
    "gradient_accumulation_steps": accumilation_steps,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "heads": heads,
        "intermediate_dim": intermediate_dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": max_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
    }
)
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.cuda.synchronize()



model = model.to(device)

loss_fn = CrossEntropyLoss()

peak_memory = 0
synchronize()

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()

loader_iter = iter(loader)
for step in range(total_steps):
    optimizer.zero_grad()
    total_loss=0
    #fp32
    """for i in range(accumilation_steps):

        batch = next(loader_iter)
        batch = batch.to(device)

        inputs = batch[:, :-1]
        target = batch[:, 1:]

        mask = causal_mask(inputs.size(1)).to(device)
        logits = model(inputs, mask)
        actual_loss = loss_fn(logits.transpose(1,2),target)
        scaled_loss = actual_loss/accumilation_steps
        scaled_loss.backward()
        total_loss += actual_loss"""

    #bf16
    for i in range(accumilation_steps):

            batch = next(loader_iter)
            batch = batch.to(device)

            inputs = batch[:, :-1]
            target = batch[:, 1:]

            mask = causal_mask(inputs.size(1)).to(device)

            if use_amp:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                    logits = model(inputs, mask)
                    actual_loss = loss_fn(logits.transpose(1, 2),target)
            else:
                logits = model(inputs, mask)
                actual_loss = loss_fn(logits.transpose(1, 2),target)

            scaled_loss = actual_loss / accumilation_steps
            scaled_loss.backward()
            total_loss += actual_loss.item()
    grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

    optimizer.step()
    scheduler.step()
    wandb.log({
    "loss": total_loss/accumilation_steps,
    "gradient_norm": grad_norm.item(),
    "learning_rate": scheduler.get_last_lr()[0],}, step=step)
    print(
        f"step: {step}, "
        f"loss: {total_loss/accumilation_steps:.4f}, "
        f"grad_norm: {grad_norm.item():.4f}")

if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
elif device.type == "mps":
        peak_memory = torch.mps.driver_allocated_memory()
else:
        peak_memory = 0 


wandb.finish()
synchronize()
end_time = time.perf_counter()
total_time = end_time-start_time
print("Total Time: ",total_time,"Sec")
tokens_per_step = (seq_len-1)*batch_size*accumilation_steps # -1 cause we are shifting
tokens_per_sec = (tokens_per_step *total_steps)/total_time
print("Tokens per second: ",tokens_per_sec)

peak_memory_mb = peak_memory / (1024 ** 2)

print("Peak memory:", peak_memory_mb, "MB")
"""x = torch.randn(10, 10, device=device, dtype=torch.bfloat16)
print(x.dtype)"""