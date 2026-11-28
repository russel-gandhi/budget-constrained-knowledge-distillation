# Sweep student compression methods and budgets with the same training loop.
# Each run writes config, epoch history, accuracy, and size to results/runs/.
import argparse
import json
import os
import time

import torch

from src.data import load_dataset, batches
from src.models import TeacherCNN, StudentMLP
from src.losses import kd_loss
from src.solvers import (train_prox, evaluate, NoReg, L1, GroupLasso, HardBudget,
                         FixedMask, support_masks)
from src.sparsity import size_report, structured_report, prunable_weights
from src.prox import topk_masks

BUDGETS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
# lambda grids chosen from short calibration runs so that the resulting
# densities roughly span the same range as BUDGETS (50% ... 1%)
L1_LAMS = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
GROUP_LAMS = [1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 1e-2]


class Runner:
    def __init__(self, args):
        self.a = args
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = load_dataset(args.dataset, device=self.dev)
        self.run_dir = os.path.join(args.out, "runs")
        self.model_dir = os.path.join(args.out, "models")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_file = open(os.path.join(args.out, "experiments_log.txt"), "a")

        teacher = TeacherCNN().to(self.dev)
        teacher.load_state_dict(torch.load(os.path.join(args.out, f"teacher_{args.dataset}.pt"),
                                           map_location=self.dev))
        teacher.eval()
        # The teacher is frozen and students see un-augmented images, so its
        # logits are computed once and reused by every run (big speed-up).
        with torch.no_grad():
            self.tlogits = torch.cat([teacher(xb) for (xb,) in
                                      batches((self.data["train"][0],), 2000, shuffle=False)])
        self.teacher_test = evaluate(teacher, *self.data["test"])
        self.log(f"teacher test acc {self.teacher_test:.4f}")
        self.d = sum(w.numel() for w in prunable_weights(StudentMLP()))

    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    # ------------------------------------------------------------------
    def run(self, name, method, reg_fn=None, T=None, alpha=None, beta=None,
            init_from=None, prune_frac=None, finetune=True, save_model=False, seed=None, **extra):
        # Train one seeded student (or load dense_kd for one-shot pruning),
        # optionally fine-tune its fixed support, then save a run JSON.
        path = os.path.join(self.run_dir, name + ".json")
        if os.path.exists(path) and not self.a.overwrite:
            self.log(f"skip {name} (exists)")
            return
        a = self.a
        T = a.T if T is None else T
        alpha = a.alpha if alpha is None else alpha
        beta = a.beta if beta is None else beta
        seed = a.seed if seed is None else seed
        torch.manual_seed(seed)                         # identical init for every run with this seed
        model = StudentMLP().to(self.dev)
        loss_fn = lambda s, t, y: kd_loss(s, t, y, T, alpha)
        tl = self.tlogits if alpha > 0 else None
        t0 = time.time()
        rec = {"name": name, "method": method, "seed": seed, "T": T, "alpha": alpha, "beta": beta,
               "epochs": a.epochs, "ft_epochs": a.ft_epochs if finetune else 0, **extra}

        if init_from is None:
            reg = reg_fn() if reg_fn else NoReg()
            rec["history"] = train_prox(model, self.data, loss_fn, reg, a.epochs, a.lr, beta,
                                        a.batch_size, tl, self.log, name,
                                        backtrack=extra.get("backtrack", not a.no_backtrack))
            if rec["history"] and rec["history"][-1].get("diverged"):
                rec["diverged"] = True
                with open(path, "w") as f:
                    json.dump(rec, f, indent=1)
                return
        else:                                           # one-shot magnitude pruning baseline
            model.load_state_dict(torch.load(os.path.join(self.model_dir, init_from + ".pt")))
            rec["history"] = []
        if prune_frac is not None:
            ws = prunable_weights(model)
            with torch.no_grad():
                for w, m in zip(ws, topk_masks(ws, int(prune_frac * self.d))):
                    w.mul_(m)

        rec["pre_ft_test_acc"] = evaluate(model, *self.data["test"])
        rec["pre_ft_val_acc"] = evaluate(model, *self.data["val"])
        sparse = size_report(model)["density"] < 0.999
        if finetune and sparse and a.ft_epochs > 0:
            rec["ft_history"] = train_prox(model, self.data, loss_fn, FixedMask(support_masks(model)),
                                           a.ft_epochs, a.ft_lr, beta, a.batch_size, tl,
                                           self.log, name + "/ft",
                                           backtrack=extra.get("backtrack", not a.no_backtrack))
        rec["test_acc"] = evaluate(model, *self.data["test"])
        rec["val_acc"] = evaluate(model, *self.data["val"])
        rec["size"] = size_report(model)
        rec["structured"] = structured_report(model, self.data["test"][0][:2000])
        rec["seconds"] = time.time() - t0
        rec["teacher_test_acc"] = self.teacher_test
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        if save_model:
            torch.save(model.state_dict(), os.path.join(self.model_dir, name + ".pt"))
        self.log(f"==> {name}: test {rec['test_acc']:.4f} (pre-ft {rec['pre_ft_test_acc']:.4f})  "
                 f"nnz {rec['size']['nnz_weights']} ({100 * rec['size']['density']:.2f}%)  "
                 f"neurons {rec['structured']['neurons_alive']}  {rec['seconds']:.0f}s")

    # ------------------------------------------------------------------
    def main_suite(self):
        a = self.a
        self.run("dense_ce", "dense_ce", alpha=0.0)
        self.run("dense_kd", "dense_kd", save_model=True)
        for f in a.budgets:
            k = int(f * self.d)
            self.run(f"iht_kd_{f}", "iht_kd", lambda k=k: HardBudget(k, None, a.ramp), budget=f)
            self.run(f"iht_ce_{f}", "iht_ce", lambda k=k: HardBudget(k, None, a.ramp), alpha=0.0, budget=f)
            self.run(f"prune_kd_{f}", "prune_kd", init_from="dense_kd", prune_frac=f, budget=f)
        for lam in a.l1_lams:
            self.run(f"l1_kd_{lam:g}", "l1_kd", lambda lam=lam: L1(lam), lam=lam)
        for lam in a.group_lams:
            self.run(f"group_kd_{lam:g}", "group_kd", lambda lam=lam: GroupLasso(lam), lam=lam)

    def ablation_suite(self):
        a = self.a
        f = a.abl_budget
        k = int(f * self.d)
        iht = lambda ramp=a.ramp: HardBudget(k, None, ramp)
        for T in [1.0, 2.0, 4.0, 8.0, 16.0]:
            self.run(f"abl_T{T:g}", "abl_T", iht, T=T, budget=f)
        for al in [0.25, 0.5, 0.75, 0.9, 1.0]:
            self.run(f"abl_alpha{al:g}", "abl_alpha", iht, alpha=al, budget=f)
        # optimiser choices
        self.run("abl_iht_beta0", "abl_beta", iht, beta=0.0, budget=f)
        self.run("abl_iht_noramp", "abl_ramp", lambda: HardBudget(k, None, 0.0), budget=f, ramp=0.0)
        lam = a.abl_l1_lam
        self.run(f"abl_l1_beta0", "abl_beta", lambda: L1(lam), beta=0.0, lam=lam)
        self.run(f"abl_l1_beta0.9", "abl_beta", lambda: L1(lam), beta=0.9, lam=lam)
        self.run("abl_l1_nobacktrack", "abl_backtrack", lambda: L1(4e-3), lam=4e-3, backtrack=False)
        self.run("abl_group_nobacktrack", "abl_backtrack", lambda: GroupLasso(2e-3), lam=2e-3, backtrack=False)

    def seeds_suite(self):
        # Repeat matched KD/label-only runs to estimate seed-to-seed variation.
        a = self.a
        for s in a.extra_seeds:
            self.run(f"dense_ce_s{s}", "dense_ce", alpha=0.0, seed=s)
            self.run(f"dense_kd_s{s}", "dense_kd", seed=s)
            for f in a.seed_budgets:
                k = int(f * self.d)
                self.run(f"iht_kd_{f}_s{s}", "iht_kd", lambda k=k: HardBudget(k, None, a.ramp), budget=f, seed=s)
                self.run(f"iht_ce_{f}_s{s}", "iht_ce", lambda k=k: HardBudget(k, None, a.ramp),
                         alpha=0.0, budget=f, seed=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion", choices=["fashion", "mnist"])
    ap.add_argument("--suite", default="all", choices=["main", "ablation", "seeds", "all"])
    ap.add_argument("--extra-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seed-budgets", type=float, nargs="+", default=[0.2, 0.05, 0.01])
    ap.add_argument("--no-backtrack", action="store_true",
                    help="use the scheduled step size as is (no line search)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ft-lr", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--ramp", type=float, default=0.5)
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    ap.add_argument("--l1-lams", type=float, nargs="+", default=L1_LAMS)
    ap.add_argument("--group-lams", type=float, nargs="+", default=GROUP_LAMS)
    ap.add_argument("--abl-budget", type=float, default=0.02)
    ap.add_argument("--abl-l1-lam", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    r = Runner(args)
    if args.suite in ("main", "all"):
        r.main_suite()
    if args.suite in ("ablation", "all"):
        r.ablation_suite()
    if args.suite in ("seeds", "all"):
        r.seeds_suite()


if __name__ == "__main__":
    main()
