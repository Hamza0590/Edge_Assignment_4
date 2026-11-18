"""
Automated benchmark of every trained model (teacher, 3 baseline students,
3 distilled students, NAS model). Requires ONNX files (run deploy.export_onnx first).

Measured per model:
  accuracy      top-1/top-5 on the full test set, PyTorch (GPU)  and ONNX Runtime (CPU)
  size          parameters, fp32 state_dict MB, ONNX file MB
  latency       batch-1 median latency on CPU: PyTorch eager and ONNX Runtime,
                with 1 thread and 4 threads
  throughput    ONNX Runtime CPU images/s at batch 64, 4 threads
  peak memory   peak inference memory of an isolated ORT process (batch 1)
  compression   size / latency ratio vs the teacher (CIFAR-100 models)

    python -m benchmark.run_benchmark
Output: outputs/results/benchmark.csv and outputs/results/kd_vs_baseline.csv
"""
import argparse
import csv
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort
import torch

from common.data import get_loaders, load_cifar_numpy
from common.profiler import (count_params, file_size_mb, ort_profile, state_dict_size_mb,
                             torch_cpu_latency)
from common.utils import evaluate, setup_cuda
from inference.model_zoo import OUT, ZOO, available_models, load_torch_model
from inference.pipeline import EdgeClassifier


def ort_throughput(path, res, batch=64, threads=4, runs=20):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, 3, res, res).astype(np.float32)
    for _ in range(3):
        sess.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"input": x})
    return batch * runs / (time.perf_counter() - t0)


def bench_model(name, data_root, runs):
    e = ZOO[name]
    model, res, meta = load_torch_model(name)
    row = {"model": name, "arch": e["arch"], "dataset": e["dataset"], "input": f"{res}x{res}",
           "params_m": count_params(model) / 1e6, "state_dict_mb": state_dict_size_mb(model),
           "onnx_mb": file_size_mb(e["onnx"])}

    # accuracy - PyTorch on GPU (bf16 autocast disabled for exact fp32 numbers)
    _, test = get_loaders(e["dataset"], data_root, 128, eval_batch_size=1000, resolution=res)
    m = model.cuda().to(memory_format=torch.channels_last)
    ev = evaluate(m, test, amp=False)
    row["torch_top1"], row["torch_top5"] = ev["top1"], ev["top5"]
    model = m.cpu()

    # accuracy - ONNX Runtime on CPU through the unified pipeline
    x, y = load_cifar_numpy(e["dataset"], data_root, train=False)
    row["onnx_top1"] = EdgeClassifier(name, backend="onnx", threads=8, data_root=data_root).accuracy(x, y)

    shape = (1, 3, res, res)
    for th in (1, 4):
        row[f"torch_cpu_ms_t{th}"] = torch_cpu_latency(model, shape, threads=th, runs=runs)["median_ms"]
        p = ort_profile(e["onnx"], shape, threads=th, runs=runs)
        row[f"ort_cpu_ms_t{th}"] = p["median_ms"]
        if th == 1:
            row["ort_p90_ms_t1"] = p["p90_ms"]
            row["peak_mem_mb"] = p["peak_mem_mb"]
    row["ort_img_per_s_b64_t4"] = ort_throughput(e["onnx"], res)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    setup_cuda()
    names = a.models or available_models()
    rows = []
    for n in names:
        t0 = time.time()
        r = bench_model(n, a.data, a.runs)
        rows.append(r)
        print(f"{n:26s} top1 {r['torch_top1']:6.2f} (onnx {r['onnx_top1']:6.2f}) | "
              f"{r['params_m']:6.3f}M {r['onnx_mb']:7.2f}MB | ORT t1 {r['ort_cpu_ms_t1']:7.2f}ms "
              f"t4 {r['ort_cpu_ms_t4']:6.2f}ms | torch t1 {r['torch_cpu_ms_t1']:7.2f}ms | "
              f"mem {r['peak_mem_mb']:6.1f}MB | {r['ort_img_per_s_b64_t4']:7.0f} img/s ({time.time() - t0:.0f}s)")

    # ratios relative to the teacher for CIFAR-100 models
    t = next((r for r in rows if r["model"] == "teacher_resnet50"), None)
    for r in rows:
        if t is not None and r["dataset"] == "cifar100":
            r["size_ratio_vs_teacher"] = t["onnx_mb"] / r["onnx_mb"]
            r["speedup_vs_teacher_t1"] = t["ort_cpu_ms_t1"] / r["ort_cpu_ms_t1"]
        else:
            r["size_ratio_vs_teacher"] = r["speedup_vs_teacher_t1"] = ""
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    path = os.path.join(OUT, "results", "benchmark.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("saved", path)

    # KD vs baseline comparison table
    by = {r["model"]: r for r in rows}
    comp = []
    for arch in ("mobilenetv2", "squeezenet", "efficientnet_lite0"):
        b, k = by.get(f"baseline_{arch}"), by.get(f"kd_{arch}")
        if b and k:
            comp.append({"student": arch, "params_m": f"{b['params_m']:.3f}",
                         "baseline_top1": f"{b['torch_top1']:.2f}", "kd_top1": f"{k['torch_top1']:.2f}",
                         "gain_top1": f"{k['torch_top1'] - b['torch_top1']:+.2f}",
                         "baseline_top5": f"{b['torch_top5']:.2f}", "kd_top5": f"{k['torch_top5']:.2f}",
                         "teacher_top1": f"{t['torch_top1']:.2f}" if t else "",
                         "ort_cpu_ms_t1": f"{k['ort_cpu_ms_t1']:.2f}"})
    if comp:
        with open(os.path.join(OUT, "results", "kd_vs_baseline.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp[0]))
            w.writeheader()
            w.writerows(comp)
    sysinfo = {"platform": platform.platform(), "processor": platform.processor(),
               "cpu_count": os.cpu_count(), "torch": torch.__version__, "onnxruntime": ort.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    with open(os.path.join(OUT, "results", "system_info.json"), "w") as f:
        json.dump(sysinfo, f, indent=2)


if __name__ == "__main__":
    main()
