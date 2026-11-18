"""Shared training utilities: seeding, logging, evaluation, checkpointing."""
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_cuda():
    """Fast-path settings for NVIDIA GPUs (TF32 matmuls + cuDNN autotuner)."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Logger:
    """Prints to stdout and appends to a log file at the same time."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        print(line)
        sys.stdout.flush()
        self.f.write(line + "\n")
        self.f.flush()


class CSVHistory:
    """Writes one row per epoch to a CSV file (training curves)."""

    def __init__(self, path, fields):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self.fields = path, fields
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def add(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


@torch.no_grad()
def evaluate(model, loader, amp=True):
    """Top-1 / top-5 accuracy and mean CE loss of `model` on `loader`."""
    model.eval()
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        out = out.float()
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        top5 = out.topk(min(5, out.shape[1]), dim=1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(1).sum().item()
        total += y.numel()
    return {"top1": 100.0 * correct1 / total, "top5": 100.0 * correct5 / total,
            "loss": loss_sum / total}


def make_optimizer(model, lr, wd, momentum=0.9, extra_params=None):
    """SGD with Nesterov; no weight decay on BN / bias parameters."""
    decay, no_decay = [], []
    params = list(model.named_parameters())
    if extra_params is not None:
        params += list(extra_params.named_parameters())
    for n, p in params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.SGD([{"params": decay, "weight_decay": wd},
                            {"params": no_decay, "weight_decay": 0.0}],
                           lr=lr, momentum=momentum, nesterov=True)


def make_scheduler(opt, epochs, steps_per_epoch, warmup_epochs=1):
    """Linear warm-up followed by cosine decay, stepped every iteration."""
    total = epochs * steps_per_epoch
    warm = warmup_epochs * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def save_checkpoint(path, model, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        now = time.time()
        d, self.t = now - self.t, now
        return d
