"""
Re-train the best discovered architecture FROM SCRATCH (fresh random weights)
on the full 50k CIFAR-10 training set and evaluate on the 10k test set.

Called automatically at the end of search.py; can also be run on its own:

    python -m q2_nas.train_best --arch outputs/nas/best_arch.json --epochs 100
"""
import argparse
import csv
import json
import os

import torch

from common.data import get_loaders
from common.profiler import count_params
from common.utils import Logger, save_checkpoint, save_json, set_seed, setup_cuda
from q2_nas.generator import build_from_config
from q2_nas.train_utils import train_candidate


def train_best(arch_json, epochs=100, data="data", out="outputs/nas", seed=0, lr=0.05, wd=5e-4):
    setup_cuda()
    set_seed(seed)
    log = Logger(os.path.join(out, "train_best.log"))
    with open(arch_json) as f:
        best = json.load(f)
    arch = best["arch_config"]
    log(f"Re-training best architecture from scratch: {best['summary']}")
    train, test = get_loaders("cifar10", data, 128, eval_batch_size=1000,
                              resolution=arch["resolution"], seed=seed)
    model = build_from_config(arch, 10)
    log(f"params: {count_params(model) / 1e6:.3f} M, input {arch['resolution']}x{arch['resolution']}")
    hist = []
    r = train_candidate(model, train, test, epochs, lr=lr, wd=wd, log=log, tag="nas_best",
                        warmup_epochs=2, history=hist)
    ckpt = os.path.join(out, "nas_best.pth")
    save_checkpoint(ckpt, model, {"arch_config": arch, "num_classes": 10, "dataset": "cifar10",
                                  "epochs": epochs, "test_top1": r["eval_top1"],
                                  "summary": best["summary"]})
    with open(os.path.join(out, "nas_best_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    res = {"summary": best["summary"], "test_top1": r["eval_top1"], "epochs": epochs,
           "train_time_min": r["train_time_s"] / 60, "checkpoint": ckpt,
           "params_m": count_params(model) / 1e6}
    save_json(os.path.join(out, "nas_best_result.json"), res)
    log(f"final test top-1: {r['eval_top1']:.2f}%  -> {ckpt}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="outputs/nas/best_arch.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/nas")
    a = ap.parse_args()
    train_best(a.arch, a.epochs, a.data, a.out, a.seed)


if __name__ == "__main__":
    main()
