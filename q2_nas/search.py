"""
Hardware-aware evolutionary Neural Architecture Search on CIFAR-10.

Algorithm: Regularized (aging) Evolution, Real et al. 2019
    1. Initialise a population of P random *feasible* architectures.
    2. Repeat until the training budget is used:
         sample S individuals from the population (tournament),
         parent = fittest of the sample, child = mutate(parent),
         evaluate child, add it to the population and remove the OLDEST member.
   Ageing (instead of removing the worst) keeps the search exploring and is robust
   to the noise of short proxy training.

Evaluation of one candidate (hardware-aware pipeline):
    a. build the network from its config                      (generator.py)
    b. MEASURE on CPU: ONNX-Runtime latency, ONNX/state-dict size, peak memory
                                                              (hardware.py)
    c. REJECT immediately if any deployment constraint is violated (no training cost)
    d. otherwise train for a few proxy epochs on 45k CIFAR-10 images and
       measure accuracy on a 5k held-out validation split    (train_utils.py)
    e. fitness = val_acc * (latency / max_latency) ** LATENCY_W
       (MnasNet-style soft reward; with LATENCY_W < 0 faster nets get a small bonus)

Outputs (in --out):
    search_history.csv / .json  - every sampled architecture, its measurements,
                                  status (trained / rejected / duplicate) and fitness
    best_arch.json              - best feasible architecture + its metrics
After the search the best architecture is automatically re-trained from scratch
(train_best.py) on the full training set unless --final-epochs 0.

    python -m q2_nas.search --budget 60 --proxy-epochs 6
"""
import argparse
import csv
import json
import os
import random
import time

import torch

from common.data import get_loaders
from common.utils import Logger, save_json, set_seed, setup_cuda
from q2_nas.generator import build_from_config
from q2_nas.hardware import Constraints, measure_hardware
from q2_nas.search_space import (SPACE_DESCRIPTION, arch_key, arch_summary, mutate,
                                 sample_architecture, search_space_size)
from q2_nas.train_utils import train_candidate

CSV_FIELDS = ["id", "iteration", "phase", "parent_id", "status", "violations", "summary",
              "resolution", "num_stages", "total_blocks", "params_m", "onnx_size_mb",
              "state_dict_mb", "ort_latency_ms", "torch_cpu_latency_ms", "peak_mem_mb",
              "val_top1", "fitness", "train_time_s", "arch"]


class NAS:
    def __init__(self, a, log):
        self.a, self.log = a, log
        self.rng = random.Random(a.seed)
        self.cons = Constraints(a.max_latency, a.max_size, a.max_mem)
        self.history, self.cache = [], {}
        self.n_trained = 0
        self.train_loader, self.val_loader = get_loaders(
            "cifar10", a.data, a.batch_size, eval_batch_size=1000, val_split=5000, seed=a.seed)
        os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, arch, phase, parent_id=None):
        key = arch_key(arch)
        rec = {"id": len(self.history), "iteration": self.n_trained, "phase": phase,
               "parent_id": parent_id, "summary": arch_summary(arch), "arch": arch,
               "resolution": arch["resolution"], "num_stages": len(arch["stages"]),
               "total_blocks": sum(s["depth"] for s in arch["stages"])}
        if key in self.cache:                     # already evaluated: reuse its results
            rec = {**self.cache[key], "id": rec["id"], "iteration": rec["iteration"],
                   "phase": phase, "parent_id": parent_id, "status": "duplicate"}
            self._record(rec)
            return rec

        hw = measure_hardware(arch, num_classes=10, threads=self.a.threads)
        rec.update(hw)
        viol = self.cons.check(hw)
        rec["violations"] = ";".join(viol)
        if viol:
            rec.update(status="rejected", val_top1=None, fitness=None, train_time_s=0)
            self.log(f"[{rec['id']:4d}] REJECT {rec['summary']}  ({rec['violations']})")
        else:
            set_seed(self.a.seed)
            model = build_from_config(arch, 10)
            self.train_loader.res = self.val_loader.res = arch["resolution"]
            r = train_candidate(model, self.train_loader, self.val_loader, self.a.proxy_epochs,
                                lr=self.a.lr, wd=self.a.wd)
            fit = r["eval_top1"] * (hw["ort_latency_ms"] / self.cons.max_latency_ms) ** self.a.latency_w
            rec.update(status="trained", val_top1=r["eval_top1"], fitness=fit,
                       train_time_s=r["train_time_s"])
            self.n_trained += 1
            self.log(f"[{rec['id']:4d}] #{self.n_trained:3d} val {r['eval_top1']:5.2f} fit {fit:6.2f} | "
                     f"lat {hw['ort_latency_ms']:.2f}ms size {hw['onnx_size_mb']:.2f}MB "
                     f"mem {hw['peak_mem_mb']:.1f}MB | {rec['summary']} ({r['train_time_s']:.0f}s)")
            del model
            torch.cuda.empty_cache()
        self.cache[key] = {k: rec.get(k) for k in rec}
        self._record(rec)
        return rec

    def _record(self, rec):
        self.history.append(rec)
        # rewrite both files after every candidate so a crash never loses the search
        with open(os.path.join(self.a.out, "search_history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for h in self.history:
                row = {k: h.get(k) for k in CSV_FIELDS}
                row["arch"] = json.dumps(h["arch"])
                w.writerow(row)
        save_json(os.path.join(self.a.out, "search_history.json"), self.history)

    def feasible_random(self):
        """Sample random architectures until one passes the constraints and is trained."""
        for _ in range(500):
            rec = self.evaluate(sample_architecture(self.rng), "init")
            if rec["status"] == "trained":
                return rec
        raise RuntimeError("could not find a feasible architecture - constraints too tight?")

    # ------------------------------------------------------------------ search loop
    def run(self):
        a = self.a
        population = []
        while len(population) < a.population and self.n_trained < a.budget:
            population.append(self.feasible_random())
        attempts = 0
        while self.n_trained < a.budget and attempts < a.budget * 50:
            attempts += 1
            tournament = self.rng.sample(population, min(a.sample_size, len(population)))
            parent = max(tournament, key=lambda r: r["fitness"])
            child = self.evaluate(mutate(parent["arch"], self.rng), "evolve", parent["id"])
            if child["status"] != "trained":
                continue                                    # infeasible/duplicate: not added
            population.append(child)
            population.pop(0)                               # ageing: drop the oldest
        trained = [h for h in self.history if h["status"] == "trained"]
        best = max(trained, key=lambda r: r["fitness"])
        out = {"best_id": best["id"], "summary": best["summary"], "arch_config": best["arch"],
               "proxy_val_top1": best["val_top1"], "fitness": best["fitness"],
               "measurements": {k: best[k] for k in ("params_m", "onnx_size_mb", "state_dict_mb",
                                                     "ort_latency_ms", "torch_cpu_latency_ms",
                                                     "peak_mem_mb", "peak_mem_method")},
               "constraints": self.cons.to_dict(),
               "search": {"algorithm": "regularized evolution", "population": a.population,
                          "tournament": a.sample_size, "trained_budget": a.budget,
                          "proxy_epochs": a.proxy_epochs, "latency_w": a.latency_w,
                          "total_sampled": len(self.history),
                          "rejected": sum(h["status"] == "rejected" for h in self.history),
                          "duplicates": sum(h["status"] == "duplicate" for h in self.history),
                          "space_size": search_space_size(), "space": SPACE_DESCRIPTION,
                          "seed": a.seed}}
        save_json(os.path.join(a.out, "best_arch.json"), out)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60, help="number of candidates to TRAIN")
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--sample-size", type=int, default=5, help="tournament size")
    ap.add_argument("--proxy-epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-latency", type=float, default=1.5, help="ms, ORT CPU batch-1")
    ap.add_argument("--max-size", type=float, default=2.0, help="MB, fp32 ONNX file")
    ap.add_argument("--max-mem", type=float, default=15.0, help="MB, peak inference memory")
    ap.add_argument("--latency-w", type=float, default=-0.03, help="MnasNet-style latency exponent")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads used for latency")
    ap.add_argument("--final-epochs", type=int, default=100, help="0 = skip final training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/nas")
    a = ap.parse_args()

    setup_cuda()
    set_seed(a.seed)
    log = Logger(os.path.join(a.out, "search.log"))
    log(f"args: {vars(a)}")
    log(f"search-space size: {search_space_size():.3e} architectures")
    t0 = time.time()
    best = NAS(a, log).run()
    log(f"search finished in {(time.time() - t0) / 60:.1f} min")
    log(f"BEST: {best['summary']} proxy-val {best['proxy_val_top1']:.2f} "
        f"fitness {best['fitness']:.2f} {best['measurements']}")

    if a.final_epochs > 0:      # automatically retrain the winner from scratch
        from q2_nas.train_best import train_best
        train_best(os.path.join(a.out, "best_arch.json"), epochs=a.final_epochs,
                   data=a.data, out=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
