"""Quick live demo (~20-40 s on a GPU): distil the teacher into a student that
may keep only a small fraction of its weights, using KD + IHT, and print what
the solver did.

    python demo.py                 # 2% budget
    python demo.py --budget 0.01   # 1% budget
Needs results/teacher_fashion.pt (run train_teacher.py first, ~3 min).
"""
import argparse
import os
import time

import torch

from src.data import load_dataset, batches
from src.losses import kd_loss
from src.models import TeacherCNN, StudentMLP
from src.solvers import train_prox, evaluate, HardBudget, FixedMask, support_masks
from src.sparsity import size_report, structured_report, prunable_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.02, help="fraction of weights kept")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--ft-epochs", type=int, default=2)
    args = ap.parse_args()

    path = os.path.join("results", "teacher_fashion.pt")
    if not os.path.exists(path):
        raise SystemExit("No teacher found: run  python train_teacher.py  first (~3 min).")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset("fashion", device=dev)

    teacher = TeacherCNN().to(dev)
    teacher.load_state_dict(torch.load(path, map_location=dev))
    teacher.eval()
    with torch.no_grad():
        tl = torch.cat([teacher(xb) for (xb,) in batches((data["train"][0],), 2000, shuffle=False)])
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params, "
          f"test acc {100 * evaluate(teacher, *data['test']):.2f}%")

    torch.manual_seed(0)
    student = StudentMLP().to(dev)
    d = sum(w.numel() for w in prunable_weights(student))
    k = int(args.budget * d)
    print(f"student: {d:,} weights, budget k = {k:,} ({100 * args.budget:g}%)\n")

    loss = lambda s, t, y: kd_loss(s, t, y, T=4.0, alpha=0.9)
    t0 = time.time()
    train_prox(student, data, loss, HardBudget(k, None, ramp=0.5), args.epochs, 0.05, 0.9,
               128, tl, print, "KD+IHT")
    train_prox(student, data, loss, FixedMask(support_masks(student)), args.ft_epochs, 0.01, 0.9,
               128, tl, print, "fine-tune")

    s = size_report(student)
    st = structured_report(student, data["test"][0][:2000])
    print(f"\ndone in {time.time() - t0:.0f}s")
    print(f"test accuracy      : {100 * evaluate(student, *data['test']):.2f}%")
    print(f"non-zero weights   : {s['nnz_weights']:,} of {d:,}  (constraint ||w||_0 <= {k:,} holds: "
          f"{s['nnz_weights'] <= k})")
    print(f"weights per layer  : {s['layer_nnz']}  of {s['layer_size']}")
    print(f"storage            : {s['sparse_kb']:.1f} KB sparse vs {s['dense_kb']:.1f} KB dense")
    print(f"neurons left       : {st['neurons_alive']} of [300, 100]")


if __name__ == "__main__":
    main()
