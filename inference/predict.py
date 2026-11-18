"""
Command-line front-end of the unified inference pipeline.

Examples
    # list models
    python -m inference.predict --list
    # classify your own image(s) with any student, PyTorch backend
    python -m inference.predict --model kd_mobilenetv2 --images cat.png dog.jpg
    # the same with ONNX Runtime on CPU
    python -m inference.predict --model kd_mobilenetv2 --backend onnx --images cat.png
    # classify CIFAR test images 0..7 with every student model
    python -m inference.predict --model all-students --cifar-test 0 8
    # NAS model (CIFAR-10)
    python -m inference.predict --model nas_best --backend onnx --cifar-test 0 8
    # full test-set accuracy
    python -m inference.predict --model kd_squeezenet --evaluate
"""
import argparse
import time

from common.data import load_cifar_numpy
from inference.model_zoo import STUDENT_MODELS, ZOO, available_models
from inference.pipeline import EdgeClassifier, load_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kd_mobilenetv2",
                    help="model name, 'all-students' or 'all'")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--images", nargs="*", help="image files to classify")
    ap.add_argument("--cifar-test", nargs=2, type=int, metavar=("START", "END"),
                    help="classify CIFAR test images [START, END)")
    ap.add_argument("--evaluate", action="store_true", help="accuracy on the full test set")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    if a.list:
        avail = set(available_models())
        for n, e in ZOO.items():
            print(f"{n:26s} {e['dataset']:9s} {'trained' if n in avail else 'missing'}")
        return

    names = (STUDENT_MODELS if a.model == "all-students" else
             list(ZOO) if a.model == "all" else [a.model])
    for name in names:
        clf = EdgeClassifier(name, backend=a.backend, device=a.device, data_root=a.data)
        print(f"\n=== {name}  [{a.backend}]  dataset={clf.dataset}  input={clf.resolution}x{clf.resolution}")
        if a.images:
            for p, preds in zip(a.images, clf.predict_topk(load_images(a.images), a.topk)):
                print(f"  {p}: " + ", ".join(f"{c} ({pr:.2%})" for c, pr in preds))
        if a.cifar_test:
            x, y = load_cifar_numpy(clf.dataset, a.data, train=False)
            s, e = a.cifar_test
            for i, preds in zip(range(s, e), clf.predict_topk(x[s:e], a.topk)):
                ok = "OK " if preds[0][0] == clf.classes[y[i]] else "XX "
                print(f"  {ok}#{i:5d} true={clf.classes[y[i]]:14s} pred: " +
                      ", ".join(f"{c} ({pr:.1%})" for c, pr in preds))
        if a.evaluate:
            x, y = load_cifar_numpy(clf.dataset, a.data, train=False)
            t0 = time.time()
            acc = clf.accuracy(x, y)
            print(f"  test accuracy: {acc:.2f}%  ({len(y)} images, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
