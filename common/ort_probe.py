"""
Isolated ONNX Runtime probe (run as a subprocess by common/profiler.py).

Loads ONE onnx model on the CPU execution provider, measures single-input
latency and the peak memory used by model loading + inference, and prints a
JSON line on stdout. Running in a fresh process makes the peak-memory number
attributable to this model only.
"""
import argparse
import json
import sys
import time

import numpy as np
import psutil


def peak_rss_bytes():
    """Peak resident memory of this process, where the OS supports it."""
    p = psutil.Process()
    mi = p.memory_info()
    if hasattr(mi, "peak_wset"):                 # Windows: peak working set
        return mi.peak_wset, "peak_wset"
    try:                                          # Linux / macOS
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (r if sys.platform == "darwin" else r * 1024), "ru_maxrss"
    except Exception:
        return None, "unsupported"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("--shape", type=int, nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()

    import onnxruntime as ort
    x = np.random.randn(*a.shape).astype(np.float32)
    base_rss = psutil.Process().memory_info().rss
    base_peak, method = peak_rss_bytes()

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    for _ in range(a.warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)

    peak, _ = peak_rss_bytes()
    res = {"median_ms": float(np.median(t)), "mean_ms": float(t.mean()),
           "p90_ms": float(np.percentile(t, 90)), "std_ms": float(t.std()),
           "peak_mem_method": method,
           "peak_mem_mb": None if peak is None else (peak - base_rss) / 1e6,
           "process_peak_mb": None if peak is None else peak / 1e6}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
