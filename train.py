import wandb
from data_pipeline import Token_dataset
from torch.utils.data import DataLoader
from model.transformer_block import Transformer,causal_mask
from torch.optim import AdamW,lr_scheduler
from torch.nn import CrossEntropyLoss
import numpy as np
import torch

vocab_size = 35000
d_model = 512
num_layers = 4
heads = 8
intermediate_dim = 2048
seq_len = 1024
batch_size = 2

model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    heads=heads,
    intermediate_dim=intermediate_dim
)

optimizer = AdamW(model.parameters(),lr=3e-4,betas=(0.9, 0.999),eps=1e-8,weight_decay=0.01)

warmup_steps = 20
total_steps = 100
min_lr = 1e-5
max_lr = 3e-4

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

loss_fn = CrossEntropyLoss()
for step,batch in enumerate(loader):
    if step>=100:
        break
    inputs = batch[:,:-1]
    target = batch[:,1:]
    mask = causal_mask(inputs.size(1))
    logits  = model(inputs,mask)
    loss = loss_fn(logits.transpose(1,2),target)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

    optimizer.step()
    scheduler.step()
    wandb.log({
    "loss": loss.item(),
    "gradient_norm": grad_norm.item(),
    "learning_rate": scheduler.get_last_lr()[0],}, step=step)
    print(
        f"step: {step}, "
        f"loss: {loss.item():.4f}, "
        f"grad_norm: {grad_norm.item():.4f}"
    )
    optimizer.zero_grad()

wandb.finish()