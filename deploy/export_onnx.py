"""
Export trained models to ONNX and verify them.

    python -m deploy.export_onnx                 # every trained model in the zoo
    python -m deploy.export_onnx --models kd_mobilenetv2 nas_best

For each model:
  1. load the PyTorch checkpoint, export with a dynamic batch dimension (opset 17)
  2. onnx.checker validates the graph
  3. ONNX Runtime (CPU) output is compared with the PyTorch output on random
     inputs (max absolute logit difference must be < 1e-3)
Results go to outputs/results/onnx_export.csv
"""
import argparse
import csv
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common.profiler import export_onnx, file_size_mb
from inference.model_zoo import OUT, ZOO, available_models, load_torch_model


def export_and_verify(name):
    model, res, _ = load_torch_model(name)
    path = ZOO[name]["onnx"]
    export_onnx(model, path, (1, 3, res, res))
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(8, 3, res, res).astype(np.float32)          # batch 8 tests dynamic axis
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - out).max())
    return {"model": name, "onnx_path": os.path.relpath(path, os.path.dirname(OUT)),
            "input_shape": f"Nx3x{res}x{res}", "onnx_size_mb": round(file_size_mb(path), 3),
            "max_abs_diff": f"{diff:.2e}", "same_argmax": bool((ref.argmax(1) == out.argmax(1)).all()),
            "verified": diff < 1e-3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    names = a.models or available_models()
    rows = []
    for n in names:
        r = export_and_verify(n)
        rows.append(r)
        print(f"{n:26s} -> {r['onnx_path']:38s} {r['onnx_size_mb']:8.3f} MB  "
              f"max|diff|={r['max_abs_diff']}  verified={r['verified']}")
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    with open(os.path.join(OUT, "results", "onnx_export.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
