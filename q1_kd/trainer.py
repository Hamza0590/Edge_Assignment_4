"""
Generic training loop shared by teacher, baseline-student and KD-student training.

* bf16 autocast + channels_last for speed on the GPU
* SGD + Nesterov, linear warm-up + cosine LR decay (per-iteration)
* evaluates on the test set every epoch (for curves only), saves the
  final-epoch checkpoint (no test-set model selection),
  logs every epoch to a CSV (training curves) and a text log.
"""
import os
import time

import torch
import torch.nn.functional as F

from common.utils import (CSVHistory, Logger, evaluate, make_optimizer,
                          make_scheduler, save_checkpoint, save_json)


def train_model(model, train_loader, test_loader, *, epochs, lr, wd, run_name,
                out_dir, meta, distiller=None, log=None, warmup_epochs=1):
    """Train `model`. If `distiller` is given, the loss comes from the distiller
    (KD); otherwise plain cross-entropy (teacher / baseline)."""
    log = log or Logger(os.path.join(out_dir, "logs", f"{run_name}.log"))
    ckpt_path = os.path.join(out_dir, "checkpoints", f"{run_name}.pth")
    hist = CSVHistory(os.path.join(out_dir, "results", "curves", f"{run_name}.csv"),
                      ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_top1",
                       "test_top5", "hard", "soft", "feat", "epoch_time_s"])

    extra = distiller.hint if (distiller is not None and distiller.hint is not None) else None
    opt = make_optimizer(model, lr, wd, extra_params=extra)
    sched = make_scheduler(opt, epochs, len(train_loader), warmup_epochs)
    runner = distiller if distiller is not None else model

    best, best_epoch = 0.0, -1
    t_start = time.time()
    for ep in range(1, epochs + 1):
        runner.train()
        t0 = time.time()
        tot_loss = torch.zeros((), device="cuda")
        tot_correct = torch.zeros((), device="cuda")
        comp = {"hard": torch.zeros((), device="cuda"), "soft": torch.zeros((), device="cuda"),
                "feat": torch.zeros((), device="cuda")}
        n = 0
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if distiller is not None:
                    loss, logits, parts = distiller(x, y)
                    for k in comp:
                        comp[k] += parts[k] * y.numel()
                else:
                    logits = model(x).float()
                    loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            # accumulate on-GPU (no .item() per step => no CPU/GPU sync)
            tot_loss += loss.detach() * y.numel()
            tot_correct += (logits.detach().argmax(1) == y).sum()
            n += y.numel()
        train_loss, train_acc = tot_loss.item() / n, 100.0 * tot_correct.item() / n
        ev = evaluate(model, test_loader)
        dt = time.time() - t0
        if ev["top1"] > best:
            best, best_epoch = ev["top1"], ep
        # We keep the FINAL-epoch weights (not the best-on-test epoch) so that the
        # test set is never used for model selection; best is only logged.
        save_checkpoint(ckpt_path, model, {**meta, "epoch": ep, "test_top1": ev["top1"],
                                           "test_top5": ev["top5"]})
        hist.add({"epoch": ep, "lr": f"{opt.param_groups[0]['lr']:.5f}",
                  "train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.2f}",
                  "test_loss": f"{ev['loss']:.4f}", "test_top1": f"{ev['top1']:.2f}",
                  "test_top5": f"{ev['top5']:.2f}",
                  "hard": f"{comp['hard'].item() / n:.4f}", "soft": f"{comp['soft'].item() / n:.4f}",
                  "feat": f"{comp['feat'].item() / n:.4f}", "epoch_time_s": f"{dt:.1f}"})
        extra_s = ""
        if distiller is not None:
            extra_s = (f" | hard {comp['hard'].item() / n:.3f} soft {comp['soft'].item() / n:.3f}"
                       f" feat {comp['feat'].item() / n:.3f}")
        log(f"[{run_name}] ep {ep:3d}/{epochs} | loss {train_loss:.3f} acc {train_acc:5.2f}"
            f" | test top1 {ev['top1']:5.2f} top5 {ev['top5']:5.2f} (best {best:5.2f}@{best_epoch})"
            f"{extra_s} | {dt:.1f}s")

    summary = {"run": run_name, **meta, "best_top1": best, "best_epoch": best_epoch,
               "final_top1": ev["top1"], "final_top5": ev["top5"],
               "train_time_min": (time.time() - t_start) / 60, "checkpoint": ckpt_path}
    save_json(os.path.join(out_dir, "results", "runs", f"{run_name}.json"), summary)
    log(f"[{run_name}] done: best top1 {best:.2f} @ epoch {best_epoch}, "
        f"{summary['train_time_min']:.1f} min -> {ckpt_path}")
    return summary
