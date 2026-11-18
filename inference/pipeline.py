"""
Unified inference pipeline for ALL trained student models (+ teacher and NAS model).

The same `EdgeClassifier` API works for every model and for two backends:
    backend="torch" : PyTorch eager (CPU or CUDA)
    backend="onnx"  : ONNX Runtime, CPU execution provider

    clf = EdgeClassifier("kd_mobilenetv2", backend="onnx")
    clf.predict_topk(images_uint8_nhwc, k=5)   ->  [[(class_name, prob), ...], ...]

Pre-processing (resize to the model's input resolution, scale to [0,1],
per-channel normalisation) is identical to training and is chosen
automatically from the model zoo entry.
"""
import numpy as np
import torch
import torchvision

from common.data import CIFAR_STATS, preprocess_numpy
from inference.model_zoo import ZOO, load_torch_model


def class_names(dataset, root="data"):
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root, train=False, download=True).classes


class EdgeClassifier:
    def __init__(self, name, backend="torch", device="cpu", threads=None, data_root="data"):
        if name not in ZOO:
            raise KeyError(f"unknown model {name}; choose from {list(ZOO)}")
        self.name, self.backend, self.device = name, backend, device
        self.entry = ZOO[name]
        self.dataset = self.entry["dataset"]
        self.classes = class_names(self.dataset, data_root)
        if backend == "torch":
            self.model, self.resolution, self.meta = load_torch_model(name)
            self.model.to(device)
            if threads:
                torch.set_num_threads(threads)
        elif backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(self.entry["onnx"], so, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.resolution = self.sess.get_inputs()[0].shape[-1]
        else:
            raise ValueError("backend must be 'torch' or 'onnx'")

    def preprocess(self, images_uint8_nhwc):
        return preprocess_numpy(images_uint8_nhwc, self.dataset, self.resolution)

    def logits(self, images_uint8_nhwc, batch_size=256):
        x = self.preprocess(images_uint8_nhwc)
        outs = []
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            if self.backend == "torch":
                with torch.no_grad():
                    outs.append(self.model(torch.from_numpy(xb).to(self.device)).float().cpu().numpy())
            else:
                outs.append(self.sess.run(None, {self.input_name: xb})[0])
        return np.concatenate(outs)

    def predict_proba(self, images_uint8_nhwc):
        z = self.logits(images_uint8_nhwc)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict_topk(self, images_uint8_nhwc, k=5):
        p = self.predict_proba(images_uint8_nhwc)
        idx = np.argsort(-p, axis=1)[:, :k]
        return [[(self.classes[j], float(p[i, j])) for j in idx[i]] for i in range(len(p))]

    def accuracy(self, images_uint8_nhwc, labels):
        pred = self.logits(images_uint8_nhwc).argmax(1)
        return 100.0 * float((pred == np.asarray(labels)).mean())


def load_images(paths):
    """Read image files with PIL -> uint8 NHWC array (resized to 32x32 like CIFAR)."""
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR))
                     for p in paths])


__all__ = ["EdgeClassifier", "load_images", "class_names", "CIFAR_STATS"]
