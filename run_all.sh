#!/usr/bin/env bash
# Full pipeline for Assignment 4 (run from the project root).
set -e
mkdir -p outputs/logs
L=outputs/logs

# ---------------- Question 1 ----------------
python -u -m q1_kd.train_teacher --epochs 60                         2>&1 | tee $L/console_teacher.txt
python -u -m q1_kd.train_baseline --arch all --epochs 100            2>&1 | tee $L/console_baseline.txt
python -u -m q1_kd.train_kd --arch all --epochs 100 --alpha 0.9 --T 4 --beta 1.0 2>&1 | tee $L/console_kd.txt
python -u -m q1_kd.train_kd --arch all --epochs 100 --beta 0 --tag kd_softonly   2>&1 | tee $L/console_kd_softonly.txt

# ---------------- Question 2 ----------------
python -u -m q2_nas.search --budget 60 --proxy-epochs 6 --final-epochs 100 2>&1 | tee $L/console_nas.txt

# ---------------- Deployment / benchmark ----------------
python -u -m deploy.export_onnx                                      2>&1 | tee $L/console_export.txt
python -u -m benchmark.run_benchmark                                 2>&1 | tee $L/console_benchmark.txt
python -u -m inference.predict --model all --backend onnx --cifar-test 0 5 2>&1 | tee $L/console_inference.txt
python -u deploy/onnx_infer.py --onnx outputs/onnx/kd_mobilenetv2.onnx --dataset cifar100 --evaluate 2>&1 | tee $L/console_onnx_infer.txt
python -u deploy/onnx_infer.py --onnx outputs/onnx/nas_best.onnx --dataset cifar10 --evaluate      2>&1 | tee -a $L/console_onnx_infer.txt
python -m report.make_figures
