"""
Baseline pipeline: train student networks on CIFAR-100 WITHOUT distillation
(plain cross-entropy on the hard labels, random initialisation).

    python -m q1_kd.train_baseline --arch all --epochs 60
    python -m q1_kd.train_baseline --arch mobilenetv2

Uses exactly the same data pipeline, optimiser, schedule, epochs and seed as
train_kd.py so the only difference between the two is the KD loss.
"""
import argparse

import torch

from common.data import get_loaders
from common.utils import Logger, set_seed, setup_cuda
from q1_kd.models import STUDENTS, build_model
from q1_kd.trainer import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    setup_cuda()

    archs = list(STUDENTS) if a.arch == "all" else [a.arch]
    for arch in archs:
        set_seed(a.seed)
        run = f"baseline_{arch}"
        log = Logger(f"{a.out}/logs/{run}.log")
        log(f"args: {vars(a)} arch={arch}")
        train, test = get_loaders("cifar100", a.data, a.batch_size, seed=a.seed)
        model = build_model(arch, 100).cuda().to(memory_format=torch.channels_last)
        train_model(model, train, test, epochs=a.epochs, lr=a.lr, wd=a.wd, run_name=run,
                    out_dir=a.out, log=log,
                    meta={"arch": arch, "dataset": "cifar100", "mode": "baseline",
                          "epochs": a.epochs, "lr": a.lr, "wd": a.wd})


if __name__ == "__main__":
    main()
