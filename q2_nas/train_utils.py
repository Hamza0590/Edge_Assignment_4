"""Training / validation routine for NAS candidates (and for the final model)."""
import time

import torch
import torch.nn.functional as F

from common.utils import evaluate, make_optimizer, make_scheduler


def train_candidate(model, train_loader, eval_loader, epochs, lr=0.05, wd=5e-4,
                    log=None, tag="", warmup_epochs=1, history=None):
    """Train with SGD + cosine schedule; return final-epoch eval accuracy.
    The loaders' `res` attribute must already be set to the candidate's input resolution."""
    model = model.cuda().to(memory_format=torch.channels_last)
    opt = make_optimizer(model, lr, wd)
    sched = make_scheduler(opt, epochs, len(train_loader), warmup_epochs=min(warmup_epochs, epochs))
    t0 = time.time()
    ev = None
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        n = 0
        te = time.time()
        for x, y in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x).float()
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.detach() * y.numel()
            correct += (out.detach().argmax(1) == y).sum()
            n += y.numel()
        ev = evaluate(model, eval_loader)
        row = {"epoch": ep, "train_loss": tot_loss.item() / n, "train_acc": 100 * correct.item() / n,
               "eval_top1": ev["top1"], "eval_loss": ev["loss"], "epoch_time_s": time.time() - te}
        if history is not None:
            history.append(row)
        if log is not None:
            log(f"  {tag} ep {ep}/{epochs} loss {row['train_loss']:.3f} train {row['train_acc']:.2f}"
                f" eval {ev['top1']:.2f} ({row['epoch_time_s']:.1f}s)")
    return {"eval_top1": ev["top1"], "eval_loss": ev["loss"], "train_time_s": time.time() - t0}
