"""
Hardware profiling helpers (used by the Q1 benchmark and by the Q2 hardware-aware NAS).

All measurements are *actual measurements* on this machine's CPU:
  * storage size   - bytes of the serialized fp32 state_dict / ONNX file
  * CPU latency    - wall-clock time of single-image (batch=1) inference,
                     after warm-up, reported as median / mean / p90 over N runs
  * peak memory    - measured in a *fresh subprocess* that only loads the ONNX
                     model with ONNX Runtime and runs inference. Peak resident set
                     (Windows: peak working set, Linux: ru_maxrss) minus the
                     baseline before the session was created. A fresh process is
                     used so allocator caches from earlier models can't hide memory.
"""
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model):
    """Size of the fp32 checkpoint (state_dict) in MB, measured by serialising it."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def file_size_mb(path):
    total = os.path.getsize(path)
    data = path + ".data"           # external-data file, if the exporter created one
    if os.path.exists(data):
        total += os.path.getsize(data)
    return total / 1e6


def _stats(times_ms):
    t = np.asarray(times_ms)
    return {"median_ms": float(np.median(t)), "mean_ms": float(t.mean()),
            "p90_ms": float(np.percentile(t, 90)), "std_ms": float(t.std())}


@torch.no_grad()
def torch_cpu_latency(model, input_shape, threads=1, warmup=10, runs=50):
    """Median batch-N latency of a PyTorch model on CPU (fp32, eager mode)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    m = model.to("cpu").eval()
    x = torch.randn(*input_shape)
    for _ in range(warmup):
        m(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        times.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(prev)
    return _stats(times)


def export_onnx(model, path, input_shape, opset=17):
    """Export to ONNX with a dynamic batch dimension (legacy TorchScript exporter,
    which produces a single self-contained .onnx file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    m = model.to("cpu").eval()
    x = torch.randn(*input_shape)
    torch.onnx.export(m, (x,), path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


def ort_profile(onnx_path, input_shape, threads=1, warmup=10, runs=50, timeout=300):
    """Run common/ort_probe.py in a fresh process: ORT CPU latency + peak memory."""
    cmd = [sys.executable, "-m", "common.ort_probe", onnx_path,
           "--shape", *[str(s) for s in input_shape],
           "--threads", str(threads), "--warmup", str(warmup), "--runs", str(runs)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ort_probe failed:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])
