"""Step 1: train the large teacher CNN once and save it.

The teacher is not the subject of the optimisation study, so it is trained
with a standard library optimiser (SGD + Nesterov momentum + weight decay)
and data augmentation, purely to get an accurate model to distil from.

    python train_teacher.py --dataset fashion --epochs 15
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from src.data import load_dataset, batches, augment
from src.models import TeacherCNN
from src.solvers import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion", choices=["fashion", "mnist"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(args.dataset, device=dev)
    model = TeacherCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"teacher parameters: {n_params:,}  device: {dev}")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    xtr, ytr = data["train"]
    total = args.epochs * math.ceil(len(ytr) / args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    flip = args.dataset == "fashion"          # flipping digits would change their meaning
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for xb, yb in batches((xtr, ytr), args.batch_size):
            loss = F.cross_entropy(model(augment(xb, flip=flip)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1:2d}  loss {loss.item():.4f}  val acc {evaluate(model, *data['val']):.4f}")

    res = {
        "dataset": args.dataset,
        "params": n_params,
        "val_acc": evaluate(model, *data["val"]),
        "test_acc": evaluate(model, *data["test"]),
        "train_seconds": time.time() - t0,
    }
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, f"teacher_{args.dataset}.pt"))
    with open(os.path.join(args.out, f"teacher_{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(res)


if __name__ == "__main__":
    main()
