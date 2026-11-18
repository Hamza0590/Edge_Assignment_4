# dem.md: Demo notes, issues faced and how they were solved

This is a quick briefing for the demo and viva. The full write-up is in `REPORT.md`. Every choice made without your input is in `decisions/README.md`.

---

## Final results (everything is trained and in `outputs/`)

| Model | Top-1 | Params | ONNX | ORT CPU latency (1 thread) |
|---|---|---|---|---|
| Teacher ResNet-50 (CIFAR-100) | 82.93 % | 23.7 M | 94.7 MB | 19.1 ms |
| MobileNetV2: baseline → KD | 75.27 → **78.79** (+3.52) | 2.35 M | 9.4 MB | ~1.6–1.8 ms |
| SqueezeNet: baseline → KD | 72.88 → **76.00** (+3.12) | 0.78 M | 3.1 MB | 1.05 ms |
| EfficientNet-Lite0: baseline → KD | 76.12 → **79.65** (+3.53) | 3.50 M | 14.0 MB | 2.0 ms |
| NAS best (CIFAR-10) | **92.49 %** | 0.29 M | 1.15 MB | 0.89 ms, 9 MB peak mem |

* **Ablation (soft-target KD only, β = 0):** 78.47 / 75.86 / 79.08. Soft targets give about +3 points, and the feature hint adds +0.1 to +0.6.
* **NAS:** 93 architectures sampled, 60 trained, 26 rejected by the hardware constraints, 7 duplicates. The search took 42 minutes.
* **ONNX:** all eight exports verified, max |Δlogit| ≤ 1.1e-5. ONNX Runtime accuracy is within ±0.03 points of PyTorch.

---

## Issues faced and solutions

### Issue 1: The GPU sat idle because data loading was the bottleneck (Windows)
**Problem.** On Windows, the standard `DataLoader` + PIL transform pipeline starts its worker processes with *spawn*, not *fork*. Each 32×32 image then goes through Python-level `RandomCrop`/`RandomHorizontalFlip`. For small CIFAR models this is well known to leave the GPU waiting on the CPU. I identified this up front and did not benchmark the slow path. On top of that, the course needs about 11 training runs plus 60+ NAS candidates.
**Solution.** `common/data.py` loads all of CIFAR onto the GPU once as uint8 (about 150 MB). It then does the augmentation in batches on the GPU: reflect-pad, a random crop through index gathering, and a random flip through `torch.where`. Together with bf16 autocast and channels-last, one MobileNetV2 epoch takes about 8 s when it runs alone, and a ResNet-50 epoch about 23 s. That made it possible to train everything in a single night.
**Viva point:** the augmentation is exactly the standard CIFAR recipe (pad 4 + crop 32 + flip). Only *where* it runs changed.

### Issue 2: ImageNet ResNet-50 does not fit 32×32 images, and feature maps had to be aligned for feature KD
**Problem.** The ImageNet stem (7×7 stride-2 conv + max-pool) turns a 32×32 image into 8×8 before the first block, and the network would end at 1×1. The students use different channel counts at their last feature map (MobileNetV2 1280, SqueezeNet 512, EfficientNet-Lite0 1280, teacher 2048), so feature distillation cannot compare them directly.
**Solution.**
* Every network uses a 3×3 stride-1 stem and downsamples 8× in total. **All four models end in a 4×4 map.**
* The teacher keeps the ImageNet-pretrained residual stages. Only the new stem and the classifier start from scratch. It reached about 74% after 3 epochs and **83.05%** after 60.
* For feature KD, a 1×1 conv + BN *regressor* maps the student's channels to 2048. The loss compares L2-normalised feature tensors, so it stays between 0 and 4 and cannot swamp the CE/KL terms. Raw-feature MSE would scale with the activation size of each network. A sanity check gave about 2.0 at initialisation, falling to about 0.7–0.85 during training. The regressor is thrown away after training, so the deployed student is unchanged.

### Issue 3: Measuring "peak inference memory" reliably
**Problem.** Inside the training process the peak RSS is a process-wide *maximum*. It already contains the dataset, CUDA context and every earlier model, and the CPU allocator keeps freed memory cached. A per-model number taken there is meaningless.
**Solution.** `common/ort_probe.py` runs **each model in a fresh subprocess** that imports only onnxruntime. It records RSS before creating the session and the OS peak after inference (Windows `peak_wset`, Linux `ru_maxrss`), and the difference is the model's peak inference memory. When the OS has no peak counter the value is `None` and the memory constraint is skipped. This is the "where supported" in the brief. The same probe gives the ONNX Runtime latency, so the NAS constraints are checked against the **real deployment runtime**.

### Issue 4: Latency measurements were noisy, and an ONNX accuracy mismatch appeared
**Problem.** (a) The NAS ran at the same time as the KD training jobs, which also use the CPU, so single latency samples jumped around. (b) During a smoke test the ONNX model scored 50.6% while PyTorch scored 52.4% on the "same" model.
**Solution.** (a) Latency is the **median of 50 runs after 10 warm-up runs**, measured with **one fixed thread**, in an isolated process. All final numbers in `benchmark.csv` were measured again with no training running. (b) The ONNX file had been exported from an older checkpoint while training was still writing new ones. `deploy/export_onnx.py` now runs only after training, and every export is checked against PyTorch on the spot (max |Δlogits| ~ 1e-6, same arg-max). `benchmark.csv` reports PyTorch and ONNX accuracy side by side as a second check.

### Smaller issues
* **SqueezeNet without BatchNorm** tends to stay at chance with lr 0.1. BN was added (decision D3), and it is folded into the convs when exporting to ONNX.
* **Slow dataset download** (CIFAR-10 took about 15 min). The code was written while it downloaded.
* **Exporter choice:** PyTorch 2.10 defaults to the dynamo ONNX exporter, which can write an external `.data` file. `dynamo=False` (TorchScript exporter, opset 17) gives one self-contained `.onnx` file per model with a dynamic batch axis.
* **Test-set leakage:** checkpoints are the *final* epoch and the "best epoch" is only logged. The NAS uses a 45k/5k split of the *training* set and touches the test set only for the final retrained model.

---

## Other important things to know for the demo

1. **Run order:** teacher → baselines → KD → NAS → export → benchmark → figures. `run_all.sh` / `run_all.ps1` do all of it. Everything is already trained, so for a live demo you only need:
   ```bash
   python -m inference.predict --list
   python -m inference.predict --model all-students --backend onnx --cifar-test 0 5
   python -m inference.predict --model nas_best --backend onnx --cifar-test 0 5
   python deploy/onnx_infer.py --onnx outputs/onnx/kd_mobilenetv2.onnx --dataset cifar100 --evaluate
   python -m benchmark.run_benchmark --models kd_mobilenetv2 nas_best --runs 50
   ```
   A short NAS demo (about 2 minutes): `python -m q2_nas.search --budget 4 --population 3 --proxy-epochs 1 --final-epochs 1 --out outputs/nas_demo`
2. **KD loss formula** (likely viva question):
   `L = (1-α)·CE(s, y) + α·T²·KL(softmax(t/T) ‖ softmax(s/T)) + β·‖norm(r(f_s)) − norm(f_t)‖²`, with α = 0.9, T = 4, β = 1.
   * T² keeps the gradient of the soft term the same size as T changes.
   * A higher T exposes the "dark knowledge": how similar the wrong classes are to each other.
3. **NAS algorithm:** regularized (aging) evolution. Tournament of 5 from a population of 16 → mutate the best of the tournament → evaluate the child → drop the **oldest** member. Ageing keeps exploring instead of over-trusting noisy 6-epoch proxy scores.
4. **Hardware-aware selection:** each candidate is exported to ONNX and measured *before* it is trained. If it breaks latency ≤ 1.5 ms, size ≤ 2 MB or peak memory ≤ 15 MB it is **rejected without any training**. Fitness among feasible nets = `val_acc × (latency/1.5 ms)^-0.03` (a MnasNet-style soft bonus for speed).
5. **Search space size:** about 3.1 × 10¹¹ architectures (resolution × stem × 2–4 stages × per-stage depth/width/kernel/block/activation).
6. **Before submitting:**
   * Run `python make_zip.py <ROLL>_<SECTION>`. It excludes `data/` and caches, then tests the zip.
   * Take a few real screenshots of your terminal running the demo commands above. `report/figures/shot_*.png` are renderings of the real console logs and are fine as backups.
   * The teacher checkpoint is about 95 MB, so the zip is large (~305 MB, including the 3 ablation checkpoints). If Classroom complains, the ONNX copy of the teacher can be dropped (it is regenerated by `deploy/export_onnx.py`).
