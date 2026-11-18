"""
Hardware-aware evaluation of a candidate architecture (no training involved).

For every candidate we *measure* on the CPU of this machine:
  1. parameter count
  2. storage size  : size of the exported fp32 ONNX file (what is deployed) and of the
                     PyTorch state_dict
  3. CPU latency   : ONNX Runtime, CPU execution provider, batch 1, fixed thread count,
                     median of N runs after warm-up (isolated subprocess)
                     + PyTorch eager CPU latency for reference
  4. peak memory   : peak RSS increase of the isolated ORT process while loading the
                     model and running inference (Windows peak working set / Linux ru_maxrss)
and compare them against the deployment constraints. Architectures violating
any constraint are rejected *before* any GPU time is spent training them.
"""
import os
import tempfile
from dataclasses import asdict, dataclass

from common.profiler import (count_params, export_onnx, file_size_mb, ort_profile,
                             state_dict_size_mb, torch_cpu_latency)
from q2_nas.generator import build_from_config


@dataclass
class Constraints:
    max_latency_ms: float = 2.0      # ORT CPU, batch 1, 1 thread, median
    max_size_mb: float = 1.0         # fp32 ONNX file
    max_peak_mem_mb: float = 40.0    # peak inference memory (where supported)

    def check(self, m):
        """Return a list of violated constraints (empty list => feasible)."""
        v = []
        if m["ort_latency_ms"] > self.max_latency_ms:
            v.append(f"latency {m['ort_latency_ms']:.2f}>{self.max_latency_ms}ms")
        if m["onnx_size_mb"] > self.max_size_mb:
            v.append(f"size {m['onnx_size_mb']:.2f}>{self.max_size_mb}MB")
        if m["peak_mem_mb"] is not None and m["peak_mem_mb"] > self.max_peak_mem_mb:
            v.append(f"mem {m['peak_mem_mb']:.1f}>{self.max_peak_mem_mb}MB")
        return v

    def to_dict(self):
        return asdict(self)


def measure_hardware(arch, num_classes=10, threads=1, runs=50, tmp_dir=None):
    model = build_from_config(arch, num_classes).eval()
    shape = (1, 3, arch["resolution"], arch["resolution"])
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="nas_")
    onnx_path = os.path.join(tmp_dir, "candidate.onnx")
    export_onnx(model, onnx_path, shape)
    ort = ort_profile(onnx_path, shape, threads=threads, runs=runs)
    tl = torch_cpu_latency(model, shape, threads=threads, runs=30)
    return {
        "params": count_params(model),
        "params_m": count_params(model) / 1e6,
        "state_dict_mb": state_dict_size_mb(model),
        "onnx_size_mb": file_size_mb(onnx_path),
        "ort_latency_ms": ort["median_ms"],
        "ort_latency_p90_ms": ort["p90_ms"],
        "torch_cpu_latency_ms": tl["median_ms"],
        "peak_mem_mb": ort["peak_mem_mb"],
        "peak_mem_method": ort["peak_mem_method"],
    }
