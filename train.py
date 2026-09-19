import wandb
from data_pipeline import Token_dataset
from torch.utils.data import DataLoader
from model.transformer_block import Transformer,causal_mask
from torch.optim import AdamW,lr_scheduler
from torch.nn import CrossEntropyLoss
import numpy as np
import time
import torch
import os

vocab_size = 35000
d_model = 768
num_layers = 5
heads = 12
intermediate_dim = 3072
seq_len = 1024
batch_size = 16
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

warmup_steps =500
total_steps = 61706
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
num_params = sum(p.numel() for p in model.parameters())
data = np.memmap("train.bin",dtype=np.uint16,mode="r")
val_data = np.memmap("val.bin",dtype = np.uint16,mode = "r")

dataset = Token_dataset(data,seq_len)
val_dataset = Token_dataset(val_data,seq_len)

loader = DataLoader(
    dataset,
    batch_size=batch_size
)

val_loader = DataLoader(
     val_dataset,
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
        "parameters": num_params,
        "parameters_millions": num_params / 1e6,
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

@torch.no_grad()
def evaluate():
    model.eval()
    val_loss=0
    val_iter = iter(val_loader)
    for _ in range(20):
        
        val_batch = next(val_iter)
        val_batch = val_batch.to(device)

        inputs = val_batch[:, :-1]
        target = val_batch[:, 1:]

        mask = causal_mask(inputs.size(1)).to(device)

        if use_amp:
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                logits = model(inputs, mask)
                actual_loss = loss_fn(logits.transpose(1, 2),target)
        else:
            logits = model(inputs, mask)
            actual_loss = loss_fn(logits.transpose(1, 2),target)
        val_loss += actual_loss.item()
    model.train()
    wandb.log({"val_loss": val_loss/20})
    return val_loss/20

def save_checkpoint(step):
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    temp_path = "checkpoints/latest.tmp.pt"
    final_path = "checkpoints/latest.pt"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, final_path)

    print(f"Checkpoint saved at step {step}")
        
def load_checkpoint():
    path = "checkpoints/latest.pt"

    if not os.path.exists(path):
        print("No checkpoint found. Starting from scratch.")
        return 0

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint["step"]

    print(f"Resuming from step {step}")

    return step

peak_memory = 0
synchronize()

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()

start_step = load_checkpoint()
loader_iter = iter(loader)
for step in range(start_step,total_steps):
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
    "learning_rate": scheduler.get_last_lr()[0]}, step=step,)
    print(
        f"step: {step}, "
        f"loss: {total_loss/accumilation_steps:.4f}, "
        f"grad_norm: {grad_norm.item():.4f}")

    if (step + 1) % 1000 == 0:
        save_checkpoint(step + 1)
        val_loss = evaluate()
        print(f"step: {step + 1}, val_loss: {val_loss:.4f}")

    if step == total_steps - 1:
        if (step + 1) % 1000 != 0:
            save_checkpoint(total_steps)
            val_loss = evaluate()
            print(f"step: {total_steps}, val_loss: {val_loss:.4f}")
save_checkpoint(total_steps)

if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
elif device.type == "mps":
        peak_memory = torch.mps.driver_allocated_memory()
else:
        peak_memory = 0 



synchronize()
end_time = time.perf_counter()
wandb.finish()
total_time = end_time-start_time
print("Total Time: ",total_time,"Sec")
tokens_per_step = (seq_len-1)*batch_size*accumilation_steps # -1 cause we are shifting
tokens_per_sec = (tokens_per_step *total_steps)/total_time
print("Tokens per second: ",tokens_per_sec)

peak_memory_mb = peak_memory / (1024 ** 2)

print("Peak memory:", peak_memory_mb, "MB")
"""x = torch.randn(10, 10, device=device, dtype=torch.bfloat16)
print(x.dtype)"""