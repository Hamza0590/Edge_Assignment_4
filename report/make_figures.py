"""
Generate every figure used in REPORT.md from the CSV/JSON results.

    python -m report.make_figures
Figures are written to report/figures/.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)
ARCHS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
COL = {"baseline": "#7f7f7f", "kd": "#1f77b4", "kd_softonly": "#ff7f0e", "teacher": "#d62728"}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def curves():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    t = read_csv(os.path.join(OUT, "results", "curves", "teacher_resnet50.csv"))
    if t:
        axes[0].plot([int(r["epoch"]) for r in t], [float(r["test_top1"]) for r in t], c=COL["teacher"],
                     label="test top-1")
        axes[0].plot([int(r["epoch"]) for r in t], [float(r["train_acc"]) for r in t], c=COL["teacher"],
                     ls="--", label="train acc")
        axes[0].set_title("Teacher ResNet-50 (CIFAR-100)")
        axes[0].legend()
    for ax, arch in zip(axes[1:], ARCHS):
        for mode in ("baseline", "kd", "kd_softonly"):
            rows = read_csv(os.path.join(OUT, "results", "curves", f"{mode}_{arch}.csv"))
            if rows:
                ax.plot([int(r["epoch"]) for r in rows], [float(r["test_top1"]) for r in rows],
                        c=COL[mode], label={"baseline": "baseline (no KD)", "kd": "KD (soft+hard+feature)",
                                            "kd_softonly": "KD (soft+hard only)"}[mode])
        ax.set_title(arch)
        ax.legend()
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1_training_curves.png"), dpi=130)
    plt.close(fig)


def kd_bars():
    bench = {r["model"]: r for r in read_csv(os.path.join(OUT, "results", "benchmark.csv"))}
    runs = {}
    for f in os.listdir(os.path.join(OUT, "results", "runs")) if os.path.exists(
            os.path.join(OUT, "results", "runs")) else []:
        with open(os.path.join(OUT, "results", "runs", f)) as fh:
            d = json.load(fh)
        runs[d["run"]] = d["final_top1"]
    acc = {m: float(r["torch_top1"]) for m, r in bench.items()} or runs
    for r in read_csv(os.path.join(OUT, "results", "ablation_softonly.csv")):   # fp32 re-evaluation
        acc[f"kd_softonly_{r['student']}"] = float(r["kd_softonly_top1"])
    acc.update({k: v for k, v in runs.items() if k not in acc})
    modes = [m for m in ("baseline", "kd_softonly", "kd") if any(f"{m}_{a}" in acc for a in ARCHS)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / max(1, len(modes))
    for i, m in enumerate(modes):
        vals = [acc.get(f"{m}_{a}", 0) for a in ARCHS]
        xs = [j + (i - (len(modes) - 1) / 2) * w for j in range(len(ARCHS))]
        b = ax.bar(xs, vals, w, color=COL[m], label=m)
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    if "teacher_resnet50" in acc:
        ax.axhline(acc["teacher_resnet50"], c=COL["teacher"], ls="--",
                   label=f"teacher ResNet-50 ({acc['teacher_resnet50']:.1f})")
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("CIFAR-100 test top-1 (%)")
    lo = min(v for v in acc.values() if v > 0)
    ax.set_ylim(max(0, lo - 10), 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Knowledge distillation vs. baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1_kd_vs_baseline.png"), dpi=130)
    plt.close(fig)


def acc_vs_latency():
    rows = read_csv(os.path.join(OUT, "results", "benchmark.csv"))
    rows = [r for r in rows if r["dataset"] == "cifar100"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in rows:
        mode = "teacher" if r["model"].startswith("teacher") else r["model"].split("_")[0]
        mode = "kd_softonly" if r["model"].startswith("kd_softonly") else mode
        ax.scatter(float(r["ort_cpu_ms_t1"]), float(r["torch_top1"]), s=30 + 8 * float(r["onnx_mb"]),
                   c=COL.get(mode, "k"), alpha=0.8)
        ax.annotate(r["model"].replace("efficientnet_lite0", "effnet-lite0"),
                    (float(r["ort_cpu_ms_t1"]), float(r["torch_top1"])), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("ONNX Runtime CPU latency, batch 1, 1 thread (ms, log)")
    ax.set_ylabel("CIFAR-100 top-1 (%)")
    ax.set_title("Accuracy vs. CPU latency (marker size ~ model size)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1_accuracy_vs_latency.png"), dpi=130)
    plt.close(fig)


def nas_plots():
    hist = read_csv(os.path.join(OUT, "nas", "search_history.csv"))
    if not hist:
        return
    best_path = os.path.join(OUT, "nas", "best_arch.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else None
    cons = best["constraints"] if best else {"max_latency_ms": 1.5, "max_size_mb": 2.0}
    tr = [h for h in hist if h["status"] == "trained"]
    rj = [h for h in hist if h["status"] == "rejected"]

    # 1) progress
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = list(range(1, len(tr) + 1))
    acc = [float(h["val_top1"]) for h in tr]
    run_best, b = [], 0
    for h in tr:
        b = max(b, float(h["fitness"]))
        run_best.append(b)
    ax.scatter(xs, acc, c=["#2ca02c" if h["phase"] == "evolve" else "#9467bd" for h in tr], s=18,
               label="candidate val acc (purple=random init, green=mutation)")
    ax.plot(xs, run_best, c="k", label="best fitness so far")
    ax.set_xlabel("trained candidate #")
    ax.set_ylabel("proxy validation acc (%) / fitness")
    ax.set_title("Evolutionary NAS progress (CIFAR-10, proxy training)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_nas_progress.png"), dpi=130)
    plt.close(fig)

    # 2) hardware space: latency vs size, feasible box, rejected points
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.scatter([float(h["ort_latency_ms"]) for h in rj], [float(h["onnx_size_mb"]) for h in rj],
               marker="x", c="#d62728", s=20, label=f"rejected ({len(rj)})")
    sc = ax.scatter([float(h["ort_latency_ms"]) for h in tr], [float(h["onnx_size_mb"]) for h in tr],
                    c=acc, cmap="viridis", s=25, label=f"trained ({len(tr)})")
    plt.colorbar(sc, ax=ax, label="proxy val acc (%)")
    ax.axvline(cons["max_latency_ms"], c="k", ls="--", lw=1)
    ax.axhline(cons["max_size_mb"], c="k", ls="--", lw=1)
    if best:
        m = best["measurements"]
        ax.scatter([m["ort_latency_ms"]], [m["onnx_size_mb"]], marker="*", s=250, c="gold",
                   edgecolors="k", label="selected", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured ORT CPU latency (ms)")
    ax.set_ylabel("ONNX size (MB)")
    ax.set_title("Hardware-aware filtering (dashed = constraints)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter([float(h["ort_latency_ms"]) for h in tr], acc, c="#1f77b4", s=20)
    if best:
        ax.scatter([best["measurements"]["ort_latency_ms"]], [best["proxy_val_top1"]], marker="*",
                   s=250, c="gold", edgecolors="k", zorder=5)
    ax.set_xlabel("measured ORT CPU latency (ms)")
    ax.set_ylabel("proxy validation acc (%)")
    ax.set_title("Accuracy / latency of feasible candidates")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_nas_hardware.png"), dpi=130)
    plt.close(fig)

    c = read_csv(os.path.join(OUT, "nas", "nas_best_curve.csv"))
    if c:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot([int(r["epoch"]) for r in c], [float(r["eval_top1"]) for r in c], label="test top-1")
        ax.plot([int(r["epoch"]) for r in c], [float(r["train_acc"]) for r in c], ls="--", label="train acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Best NAS architecture retrained from scratch (CIFAR-10)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "q2_nas_best_training.png"), dpi=130)
        plt.close(fig)


def console_screenshot(src, dst, title, max_lines=40, width=150):
    """Render the tail of a real console log as a terminal-style PNG."""
    if not os.path.exists(src):
        return
    with open(src, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n")[:width] for ln in f if "Warning" not in ln and "warn(" not in ln]
    lines = lines[-max_lines:]
    h = 0.6 + 0.19 * len(lines)
    fig = plt.figure(figsize=(14, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 1 - 0.25 / h, f"$ {title}", color="#4ec9b0", family="monospace", fontsize=9, va="top")
    for i, ln in enumerate(lines):
        fig.text(0.01, 1 - (0.5 + 0.19 * i) / h, ln, color="#d4d4d4", family="monospace", fontsize=8, va="top")
    fig.savefig(dst, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def screenshots():
    shots = [
        ("logs/console_teacher.txt", "python -m q1_kd.train_teacher --epochs 60", "shot_teacher.png"),
        ("logs/console_baseline.txt", "python -m q1_kd.train_baseline --arch all", "shot_baseline.png"),
        ("logs/console_kd.txt", "python -m q1_kd.train_kd --arch all", "shot_kd.png"),
        ("logs/console_nas.txt", "python -m q2_nas.search", "shot_nas_search.png"),
        ("logs/console_export.txt", "python -m deploy.export_onnx", "shot_onnx_export.png"),
        ("logs/console_benchmark.txt", "python -m benchmark.run_benchmark", "shot_benchmark.png"),
        ("logs/console_inference.txt", "python -m inference.predict ...", "shot_inference.png"),
        ("logs/console_onnx_infer.txt", "python deploy/onnx_infer.py ...", "shot_onnx_infer.png"),
    ]
    for src, title, dst in shots:
        console_screenshot(os.path.join(OUT, src), os.path.join(FIG, dst), title)


if __name__ == "__main__":
    curves()
    kd_bars()
    acc_vs_latency()
    nas_plots()
    screenshots()
    print("figures written to", FIG)
