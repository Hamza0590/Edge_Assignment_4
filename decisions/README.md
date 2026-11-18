# Decision log

You were asleep while this was built, so whenever a choice was needed I picked the recommended option and wrote it down here.
Each entry gives the decision, the alternatives and the reason. You can change any of them before you submit. The last column says how much work a change would take.

| # | Decision | Chosen | To change |
|---|----------|--------|-----------|
| D1 | Teacher initialisation | ImageNet-pretrained ResNet-50 body, new CIFAR stem, fine-tuned 60 epochs | `--no-pretrained --epochs 200` (~1.5 h GPU) |
| D2 | CIFAR adaptation of the networks | 3×3 stride-1 stems, total downsampling 8× (final 4×4 map) for every model | change the strides in `q1_kd/models/*` |
| D3 | SqueezeNet with BatchNorm | BN after every conv, folded into the convs at ONNX export | remove BN from `conv_bn_relu` |
| D4 | EfficientNet-Lite0 | written from scratch (torchvision has no Lite variant): no SE, ReLU6, fixed stem/head | – |
| D5 | KD hyper-parameters | α = 0.9, T = 4, β (feature) = 1.0, hint = last student map → teacher layer-4 | CLI flags of `train_kd.py` |
| D6 | Student training budget | 100 epochs, SGD 0.1, cosine, wd 5e-4, batch 128, identical for baseline and KD | `--epochs` |
| D7 | Checkpoint selection | keep the **final-epoch** weights, never the best-on-test weights | – |
| D8 | NAS strategy | Regularized (aging) **evolution**, not Optuna | – (Optuna is not installed) |
| D9 | NAS hardware target and constraints | ONNX Runtime CPU, batch 1, **1 thread**; latency ≤ 1.5 ms, ONNX size ≤ 2 MB, peak memory ≤ 15 MB | `--max-latency --max-size --max-mem` |
| D10 | NAS fitness | `val_acc × (lat / 1.5 ms)^-0.03`, a MnasNet-style soft reward on top of the hard constraints | `--latency-w` |
| D11 | NAS budget | 60 trained candidates, 6 proxy epochs each, population 16, tournament 5; the winner is retrained for 100 epochs | flags of `search.py` |
| D12 | NAS validation data | 45k/5k split of the CIFAR-10 *train* set; the test set is used only for the final model | – |
| D13 | Peak memory measurement | fresh subprocess per model, peak RSS (Windows peak working set or Linux ru_maxrss) minus the baseline | – |
| D14 | Data pipeline | whole dataset kept on the GPU, augmentation done on the GPU (no DataLoader) | – |
| D15 | Mixed precision | bf16 autocast for training; all CPU/ONNX inference runs in fp32 | – |
| D16 | Roll number / zip name | placeholder; `make_zip.py` asks for it | `python make_zip.py 22i-XXXX_X` |
| D17 | Ablation | an extra "soft-target only" KD run (β = 0), done: 78.47 / 75.86 / 79.08 vs. 78.79 / 76.00 / 79.65 with the feature term | `outputs/checkpoints/kd_softonly_*.pth` |

---

## D1 – Teacher: ImageNet-pretrained ResNet-50
* **Options:** (a) train ResNet-50 from scratch on CIFAR-100, which takes about 200 epochs to reach ~78–79%; (b) start from torchvision's ImageNet weights and fine-tune.
* **Chosen:** (b). The residual stages come from `IMAGENET1K_V2` and the 3×3 stem and 100-way classifier are trained from scratch. Fine-tuning reached ~74% after 3 epochs.
* **Why:** a stronger teacher is the point of KD, and this one costs about 4× less GPU time. The assignment says "implement and train a teacher model using ResNet-50 on CIFAR-100". The teacher is still trained end-to-end on CIFAR-100; only the initialisation is pretrained.
* **Viva note:** be ready to explain that the 7×7/stride-2 conv and the max-pool were removed because 32×32 inputs are too small for them.

## D2 – CIFAR adaptation
ImageNet networks downsample 32×, so a 32×32 image would end up as a 1×1 map. Every network here downsamples 8× instead (32 → 4). This also gives the teacher and all three students the same 4×4 final feature map, which makes feature distillation straightforward.

## D3 – SqueezeNet with BN
The original SqueezeNet has no BatchNorm and often stays at chance accuracy when trained from scratch with a high learning rate. Adding BN after each conv makes training stable. BN is folded into the conv weights during ONNX constant folding, so the deployed model has no extra cost.

## D5 – KD hyper-parameters
These are the standard values from Hinton et al. (T = 4, α = 0.9). The feature term uses normalised features, so its value stays between 0 and 4, and β = 1 keeps it the same size as the other terms. The hint layer is the last feature map, because there the teacher and student have the same spatial size (4×4) and only the channels need a 1×1 regressor.

## D7 – Final-epoch checkpoints
CIFAR has no official validation split. Picking the "best test epoch" would report an optimistic number, so every checkpoint is the last epoch and the best epoch is only logged.

## D8 – Evolution instead of Optuna
Both options are allowed. Evolution needed no extra dependency (Optuna is not installed in the conda base env), it is easy to explain in a viva, and its mutation operator works directly on the variable-length config (2–4 stages).

## D9 – Constraints
The constraints were set by profiling 15 random architectures (latency 0.27–7.4 ms, ONNX size 0.14–6.8 MB, peak memory 7–27 MB). Constraints in the middle of those ranges reject about half of the sampled architectures, which shows the constraint mechanism doing something. 1 thread was used because it gives stable timings and is a realistic edge setting (a single Cortex-A core).

## D13 – Peak memory
Measuring memory inside a long-running training process mixes the allocations of every earlier model. A fresh ONNX Runtime process per model makes the number belong to that model alone. "Where supported": on Windows this is the peak working set and on Linux it is `ru_maxrss`. On any other OS the value is recorded as unsupported (None) and the memory constraint is skipped.

## D14 – GPU data pipeline
See `dem.md`, Issue 1.
