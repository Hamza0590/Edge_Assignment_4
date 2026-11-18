"""
Model zoo: one registry describing every trained model of the project, so the
inference pipeline, the ONNX exporter and the benchmark all use the same names.

    teacher_resnet50                         (CIFAR-100, Q1 teacher)
    baseline_{mobilenetv2,squeezenet,efficientnet_lite0}   (CIFAR-100, no KD)
    kd_{mobilenetv2,squeezenet,efficientnet_lite0}         (CIFAR-100, distilled)
    nas_best                                 (CIFAR-10, Q2 best NAS architecture)
"""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
STUDENT_ARCHS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def _entry(name, arch, dataset, ckpt, num_classes):
    return {"name": name, "arch": arch, "dataset": dataset, "num_classes": num_classes,
            "checkpoint": ckpt, "onnx": os.path.join(OUT, "onnx", f"{name}.onnx")}


ZOO = {"teacher_resnet50": _entry("teacher_resnet50", "resnet50", "cifar100",
                                  os.path.join(OUT, "checkpoints", "teacher_resnet50.pth"), 100)}
for _mode in ("baseline", "kd"):
    for _a in STUDENT_ARCHS:
        _n = f"{_mode}_{_a}"
        ZOO[_n] = _entry(_n, _a, "cifar100", os.path.join(OUT, "checkpoints", f"{_n}.pth"), 100)
ZOO["nas_best"] = _entry("nas_best", "nas", "cifar10", os.path.join(OUT, "nas", "nas_best.pth"), 10)

STUDENT_MODELS = [n for n in ZOO if n.startswith(("baseline_", "kd_"))]


def available_models():
    return [n for n, e in ZOO.items() if os.path.exists(e["checkpoint"])]


def load_torch_model(name, map_location="cpu"):
    """Build the network for `name` and load its trained weights.
    Returns (model in eval mode, input resolution, meta dict)."""
    e = ZOO[name]
    ck = torch.load(e["checkpoint"], map_location=map_location, weights_only=False)
    meta = ck.get("meta", {})
    if e["arch"] == "nas":
        from q2_nas.generator import build_from_config
        model = build_from_config(meta["arch_config"], e["num_classes"])
        res = meta["arch_config"]["resolution"]
    else:
        from q1_kd.models import build_model
        kw = {"pretrained": False} if e["arch"] == "resnet50" else {}
        model = build_model(e["arch"], e["num_classes"], **kw)
        res = 32
    model.load_state_dict(ck["state_dict"])
    return model.eval(), res, meta
