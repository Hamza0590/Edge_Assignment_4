"""
Knowledge-distillation pipeline: train student networks on CIFAR-100 with a
frozen ResNet-50 teacher.

    python -m q1_kd.train_kd --arch all --epochs 60 --alpha 0.9 --T 4 --beta 1.0

  --alpha  weight of the soft (teacher) term; (1 - alpha) goes to the hard-label CE
  --T      softmax temperature
  --beta   weight of the feature-distillation (hint) term; 0 disables it
           (=> pure Hinton soft-target KD)
"""
import argparse

import torch

from common.data import get_loaders
from common.utils import Logger, set_seed, setup_cuda
from q1_kd.kd.distiller import Distiller
from q1_kd.models import STUDENTS, build_model
from q1_kd.trainer import train_model


def load_teacher(path):
    teacher = build_model("resnet50", 100, pretrained=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ck["state_dict"])
    return teacher, ck["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    ap.add_argument("--teacher", default="outputs/checkpoints/teacher_resnet50.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--tag", default="kd", help="run-name prefix (e.g. kd, kd_softonly)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    setup_cuda()

    teacher, tmeta = load_teacher(a.teacher)
    teacher = teacher.cuda().to(memory_format=torch.channels_last).eval()
    archs = list(STUDENTS) if a.arch == "all" else [a.arch]
    for arch in archs:
        set_seed(a.seed)
        run = f"{a.tag}_{arch}"
        log = Logger(f"{a.out}/logs/{run}.log")
        log(f"args: {vars(a)} arch={arch} | teacher top1={tmeta.get('test_top1')}")
        train, test = get_loaders("cifar100", a.data, a.batch_size, seed=a.seed)
        student = build_model(arch, 100).cuda().to(memory_format=torch.channels_last)
        distiller = Distiller(student, teacher, alpha=a.alpha, temperature=a.T, beta=a.beta).cuda()
        train_model(student, train, test, epochs=a.epochs, lr=a.lr, wd=a.wd, run_name=run,
                    out_dir=a.out, log=log, distiller=distiller,
                    meta={"arch": arch, "dataset": "cifar100", "mode": a.tag, "epochs": a.epochs,
                          "lr": a.lr, "wd": a.wd, "alpha": a.alpha, "T": a.T, "beta": a.beta,
                          "teacher_top1": tmeta.get("test_top1")})


if __name__ == "__main__":
    main()
