# Full pipeline for Assignment 4 (Windows PowerShell, run from the project root).
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force outputs/logs | Out-Null
$L = "outputs/logs"

# ---------------- Question 1 ----------------
python -u -m q1_kd.train_teacher --epochs 60                          | Tee-Object "$L/console_teacher.txt"
python -u -m q1_kd.train_baseline --arch all --epochs 100             | Tee-Object "$L/console_baseline.txt"
python -u -m q1_kd.train_kd --arch all --epochs 100 --alpha 0.9 --T 4 --beta 1.0 | Tee-Object "$L/console_kd.txt"
python -u -m q1_kd.train_kd --arch all --epochs 100 --beta 0 --tag kd_softonly   | Tee-Object "$L/console_kd_softonly.txt"

# ---------------- Question 2 ----------------
python -u -m q2_nas.search --budget 60 --proxy-epochs 6 --final-epochs 100 | Tee-Object "$L/console_nas.txt"

# ---------------- Deployment / benchmark ----------------
python -u -m deploy.export_onnx                                       | Tee-Object "$L/console_export.txt"
python -u -m benchmark.run_benchmark                                  | Tee-Object "$L/console_benchmark.txt"
python -u -m inference.predict --model all --backend onnx --cifar-test 0 5 | Tee-Object "$L/console_inference.txt"
python -u deploy/onnx_infer.py --onnx outputs/onnx/kd_mobilenetv2.onnx --dataset cifar100 --evaluate | Tee-Object "$L/console_onnx_infer.txt"
python -u deploy/onnx_infer.py --onnx outputs/onnx/nas_best.onnx --dataset cifar10 --evaluate | Tee-Object -Append "$L/console_onnx_infer.txt"
python -m report.make_figures
