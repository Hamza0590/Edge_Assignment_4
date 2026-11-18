# Assignment 4: Knowledge Distillation and Hardware-Aware NAS for Edge Devices

* **Q1:** Distil a ResNet-50 teacher into MobileNetV2, SqueezeNet and EfficientNet-Lite0 on CIFAR-100. Train the same students without KD as a baseline, then export them to ONNX and run them on the CPU with ONNX Runtime.
* **Q2:** Search for an edge-friendly CNN with evolutionary NAS on CIFAR-10. The search measures latency, size and memory on the real CPU and rejects architectures that break the deployment limits. The best architecture is retrained from scratch and deployed to ONNX Runtime.

The technical report is in [`REPORT.md`](REPORT.md). Design decisions are in [`decisions/`](decisions/README.md). Problems faced during development are in [`dem.md`](dem.md).

---

## 1. Installation

Tested on Windows 11, Python 3.13, PyTorch 2.10 + CUDA 12.8, on an RTX 5060 Ti (16 GB).
Any CUDA GPU will work. Training on CPU only is possible but very slow, while inference, ONNX and benchmarking run on CPU.

```bash
# (optional) fresh environment
conda create -n edge-a4 python=3.11 -y && conda activate edge-a4
# PyTorch with CUDA: pick the command for your CUDA version at https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CIFAR-10 and CIFAR-100 download into `./data` automatically on first use. The ImageNet ResNet-50 weights download automatically through torchvision.

## 2. Project layout

```
common/            shared code
  data.py          GPU-resident CIFAR loader with on-GPU augmentation
  utils.py         seeding, logging, evaluation, optimiser and LR schedule
  profiler.py      size / CPU latency / ONNX export helpers
  ort_probe.py     isolated ONNX Runtime process: latency + peak memory
q1_kd/             QUESTION 1: knowledge distillation
  models/          teacher_resnet50.py, mobilenetv2.py, squeezenet.py, efficientnet_lite.py
  kd/distiller.py  KD framework (soft targets, temperature, hard+soft, feature hints)
  trainer.py       shared training loop
  train_teacher.py teacher training script
  train_baseline.py  baseline students (no KD)
  train_kd.py      distilled students
q2_nas/            QUESTION 2: hardware-aware NAS
  search_space.py  searchable space + sampling/mutation
  generator.py     config -> nn.Module generator
  hardware.py      CPU latency / size / peak-memory measurement + constraints
  search.py        regularized-evolution search (+ automatic final training)
  train_utils.py   candidate training/validation
  train_best.py    retrain the best architecture from scratch
inference/         unified inference pipeline (all students + NAS + teacher)
  model_zoo.py, pipeline.py, predict.py
deploy/
  export_onnx.py   export + verify ONNX models
  onnx_infer.py    stand-alone ONNX Runtime CPU inference (no PyTorch needed)
benchmark/
  run_benchmark.py automated benchmark -> CSV
report/
  make_figures.py  plots + console screenshots for the report
outputs/           (generated) checkpoints, ONNX models, logs, CSV/JSON results
decisions/         decision log
run_all.sh / run_all.ps1   full pipeline in one command
make_zip.py        builds ROLLNUM_SECTION.zip with only the deliverables
```

## 3. Step-by-step execution

Run every command from the project root. Each command writes its log to `outputs/logs/`.

### Question 1: Knowledge distillation (CIFAR-100)

```bash
# 1. Teacher: ResNet-50 (ImageNet init, CIFAR stem), 60 epochs  (~25-45 min)
python -m q1_kd.train_teacher --epochs 60

# 2. Baseline students without KD (all three, 100 epochs each)
python -m q1_kd.train_baseline --arch all --epochs 100

# 3. Distilled students: soft targets (T=4) + hard labels + feature hints
python -m q1_kd.train_kd --arch all --epochs 100 --alpha 0.9 --T 4 --beta 1.0

#    (optional ablation) soft+hard KD without the feature term
python -m q1_kd.train_kd --arch all --epochs 100 --beta 0 --tag kd_softonly
```

Single student: `--arch mobilenetv2 | squeezenet | efficientnet_lite0`.

### Question 2: Hardware-aware NAS (CIFAR-10)

```bash
# Evolutionary search (60 trained candidates, 6 proxy epochs each) followed by
# AUTOMATIC re-training of the best architecture from scratch for 100 epochs
python -m q2_nas.search --budget 60 --proxy-epochs 6 \
       --max-latency 1.5 --max-size 2.0 --max-mem 15 --final-epochs 100

# (re-train the saved best architecture on its own)
python -m q2_nas.train_best --arch outputs/nas/best_arch.json --epochs 100
```

Outputs: `outputs/nas/search_history.csv|json`, `best_arch.json`, `nas_best.pth`, `search.log`.

### Deployment, inference and benchmarking

```bash
# Export every trained model to ONNX and check it against PyTorch
python -m deploy.export_onnx

# Unified inference pipeline: any model, PyTorch or ONNX Runtime backend
python -m inference.predict --list
python -m inference.predict --model all-students --backend onnx --cifar-test 0 5
python -m inference.predict --model nas_best --backend onnx --cifar-test 0 5
python -m inference.predict --model kd_mobilenetv2 --backend onnx --images my_cat.jpg
python -m inference.predict --model kd_squeezenet --backend torch --evaluate

# Stand-alone ONNX Runtime CPU inference (numpy + onnxruntime + Pillow only)
python deploy/onnx_infer.py --onnx outputs/onnx/kd_mobilenetv2.onnx --dataset cifar100 --evaluate
python deploy/onnx_infer.py --onnx outputs/onnx/nas_best.onnx --dataset cifar10 --evaluate

# Automated benchmark (accuracy, size, CPU latency, throughput, peak memory) -> CSV
python -m benchmark.run_benchmark

# Figures for the report
python -m report.make_figures
```

### Everything in one go

```bash
bash run_all.sh            # Linux / Git-Bash
powershell -File run_all.ps1   # Windows PowerShell
```

### Building the submission zip

```bash
python make_zip.py 22i-XXXX_X     # -> ../22i-XXXX_X.zip  (dataset and caches excluded)
```

## 4. Where each deliverable is

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Teacher model | `outputs/checkpoints/teacher_resnet50.pth`, `q1_kd/train_teacher.py` |
| 2 | Student architectures | `q1_kd/models/{mobilenetv2,squeezenet,efficientnet_lite}.py` |
| 3 | Baseline students | `outputs/checkpoints/baseline_*.pth`, `q1_kd/train_baseline.py` |
| 4 | Knowledge distillation | `q1_kd/kd/distiller.py`, `q1_kd/train_kd.py`, `outputs/checkpoints/kd_*.pth` |
| 5 | NAS search space | `q2_nas/search_space.py`, `q2_nas/generator.py` |
| 6 | NAS algorithm | `q2_nas/search.py`, `q2_nas/hardware.py` |
| 7 | NAS results | `outputs/nas/best_arch.json`, `outputs/nas/search_history.csv/.json` |
| 8 | Optimised architecture | `outputs/nas/nas_best.pth` |
| 9 | Inference pipeline | `inference/pipeline.py`, `inference/predict.py` |
| 10 | ONNX deployment | `outputs/onnx/*.onnx`, `deploy/export_onnx.py`, `deploy/onnx_infer.py` |
| 11 | Benchmarking | `benchmark/run_benchmark.py`, `outputs/results/benchmark.csv`, `kd_vs_baseline.csv` |
| 12 | Technical report | `REPORT.md` (+ `report/figures/`) |
| 13 | README | this file |
