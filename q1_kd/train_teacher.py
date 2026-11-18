"""
Train the ResNet-50 teacher on CIFAR-100.

    python -m q1_kd.train_teacher --epochs 30

The residual stages start from ImageNet weights (transfer learning) and are
fine-tuned end-to-end together with the new CIFAR stem and classifier.
"""
import argparse

from common.data import get_loaders
from common.utils import Logger, set_seed, setup_cuda
from q1_kd.models import build_model
from q1_kd.trainer import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true", help="train from random init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    set_seed(a.seed)
    setup_cuda()
    log = Logger(f"{a.out}/logs/teacher_resnet50.log")
    log(f"args: {vars(a)}")
    train, test = get_loaders("cifar100", a.data, a.batch_size, seed=a.seed)
    model = build_model("resnet50", 100, pretrained=not a.no_pretrained).cuda().to(
        memory_format=__import__("torch").channels_last)
    train_model(model, train, test, epochs=a.epochs, lr=a.lr, wd=a.wd,
                run_name="teacher_resnet50", out_dir=a.out, log=log,
                meta={"arch": "resnet50", "dataset": "cifar100", "mode": "teacher",
                      "epochs": a.epochs, "lr": a.lr, "wd": a.wd,
                      "pretrained_init": not a.no_pretrained})


if __name__ == "__main__":
    main()
