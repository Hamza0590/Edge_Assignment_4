"""
Stand-alone CPU inference with ONNX Runtime - depends ONLY on numpy, Pillow and
onnxruntime (no PyTorch), i.e. what would actually run on an edge device.

    # classify image files
    python deploy/onnx_infer.py --onnx outputs/onnx/kd_mobilenetv2.onnx --dataset cifar100 --images cat.png
    # accuracy + latency on the CIFAR test set (reads the python-pickle batch files directly)
    python deploy/onnx_infer.py --onnx outputs/onnx/nas_best.onnx --dataset cifar10 --evaluate
"""
import argparse
import os
import pickle
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def read_cifar_test(dataset, root="data"):
    """Read the official CIFAR test batch (python pickle format) without torchvision."""
    if dataset == "cifar10":
        path, key = os.path.join(root, "cifar-10-batches-py", "test_batch"), b"labels"
        meta, mkey = os.path.join(root, "cifar-10-batches-py", "batches.meta"), b"label_names"
    else:
        path, key = os.path.join(root, "cifar-100-python", "test"), b"fine_labels"
        meta, mkey = os.path.join(root, "cifar-100-python", "meta"), b"fine_label_names"
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    with open(meta, "rb") as f:
        names = [n.decode() for n in pickle.load(f, encoding="bytes")[mkey]]
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)       # NHWC uint8
    return x, np.asarray(d[key]), names


def preprocess(images_nhwc_uint8, dataset, res):
    """resize (PIL bilinear) -> [0,1] -> normalise -> NCHW float32"""
    if res != images_nhwc_uint8.shape[1]:
        images_nhwc_uint8 = np.stack([np.asarray(Image.fromarray(im).resize((res, res), Image.BILINEAR))
                                      for im in images_nhwc_uint8])
    mean, std = (np.asarray(v, np.float32).reshape(1, 1, 1, 3) for v in STATS[dataset])
    x = (images_nhwc_uint8.astype(np.float32) / 255.0 - mean) / std
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    res = inp.shape[-1]
    print(f"model {a.onnx} | input {inp.name} {inp.shape} | providers {sess.get_providers()}")
    _, _, names = read_cifar_test(a.dataset, a.data)

    if a.images:
        imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR))
                         for p in a.images])
        t0 = time.perf_counter()
        p = softmax(sess.run(None, {inp.name: preprocess(imgs, a.dataset, res)})[0])
        dt = (time.perf_counter() - t0) * 1e3
        for path, pi in zip(a.images, p):
            top = np.argsort(-pi)[:3]
            print(f"  {path}: " + ", ".join(f"{names[j]} ({pi[j]:.1%})" for j in top))
        print(f"  {len(imgs)} image(s) in {dt:.2f} ms")

    if a.evaluate:
        x, y, _ = read_cifar_test(a.dataset, a.data)
        correct, t_inf = 0, 0.0
        for i in range(0, len(x), a.batch_size):
            xb = preprocess(x[i:i + a.batch_size], a.dataset, res)
            t0 = time.perf_counter()
            out = sess.run(None, {inp.name: xb})[0]
            t_inf += time.perf_counter() - t0
            correct += int((out.argmax(1) == y[i:i + a.batch_size]).sum())
        print(f"  test accuracy {100.0 * correct / len(y):.2f}% on {len(y)} images | "
              f"inference {t_inf:.2f}s ({len(y) / t_inf:.0f} img/s, {a.threads} threads, batch {a.batch_size})")


if __name__ == "__main__":
    main()
