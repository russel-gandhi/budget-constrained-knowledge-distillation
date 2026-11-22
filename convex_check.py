"""Step 2: convex sanity check of the solvers.

With a *linear* (softmax-regression) student the distillation objective is
convex, so we can (i) compute the exact optimum with an off-the-shelf convex
solver (cvxpy + Clarabel) and (ii) check our proximal gradient methods against
the theory: ISTA converges as O(1/k), FISTA as O(1/k^2).

Everything here is NumPy in float64 with a hand-derived gradient (checked by
finite differences), so it does not rely on autograd at all.

Problem (theta = (W, b), W in R^{C x p}, b in R^C):
    min  f(W, b) + lam * ||W||_1
    f = (1/n) sum_i [ alpha T^2 KL(q_i || softmax(z_i / T)) + (1-alpha) CE(y_i, softmax(z_i)) ]
    z_i = W x_i + b,   q_i = softmax(teacher_logits_i / T)

    python convex_check.py --n 500 --iters 5000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as tF
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_dataset
from src.models import TeacherCNN


# ------------------------------------------------------------------ objective
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


class KDProblem:
    def __init__(self, X, Y, Q, T, alpha, lam):
        self.X, self.Y, self.Q, self.T, self.alpha, self.lam = X, Y, Q, T, alpha, lam
        self.n, self.p = X.shape
        self.C = Y.shape[1]
        # constant sum_c q log q (makes the KL term a true KL, >= 0)
        self.qlogq = np.sum(Q * np.log(np.clip(Q, 1e-300, None)), axis=1).mean()

    def f(self, W, b):
        Z = self.X @ W.T + b
        kl = self.qlogq - np.sum(self.Q * log_softmax(Z / self.T), axis=1).mean()
        ce = -np.sum(self.Y * log_softmax(Z), axis=1).mean()
        return self.alpha * self.T ** 2 * kl + (1 - self.alpha) * ce

    def grad(self, W, b):
        """d f / d z_i = [ alpha*T*(softmax(z_i/T) - q_i) + (1-alpha)*(softmax(z_i) - y_i) ] / n
        then chain rule through z = XW^T + b."""
        Z = self.X @ W.T + b
        G = (self.alpha * self.T * (softmax(Z / self.T) - self.Q)
             + (1 - self.alpha) * (softmax(Z) - self.Y)) / self.n
        return G.T @ self.X, G.sum(axis=0)

    def h(self, W):
        return self.lam * np.abs(W).sum()

    def F(self, W, b):
        return self.f(W, b) + self.h(W)

    def lipschitz(self):
        """Hessian of log-sum-exp is <= 1/2 I, and the T^2 factor exactly
        cancels the 1/T^2 from z/T, so grad f is L-Lipschitz with
        L = ||[X 1]||_2^2 / (2n)   -- independent of T and alpha."""
        Xt = np.hstack([self.X, np.ones((self.n, 1))])
        return np.linalg.norm(Xt, 2) ** 2 / (2 * self.n)


def soft(V, tau):
    return np.sign(V) * np.maximum(np.abs(V) - tau, 0.0)


# ------------------------------------------------------------------ solvers
def ista(P, L, iters):
    W, b = np.zeros((P.C, P.p)), np.zeros(P.C)
    hist = []
    for _ in range(iters):
        gW, gb = P.grad(W, b)
        W, b = soft(W - gW / L, P.lam / L), b - gb / L     # bias is not penalised
        hist.append(P.F(W, b))
    return W, b, np.array(hist)


def fista(P, L, iters):
    W, b = np.zeros((P.C, P.p)), np.zeros(P.C)
    yW, yb, t = W.copy(), b.copy(), 1.0
    hist = []
    for _ in range(iters):
        gW, gb = P.grad(yW, yb)
        Wn, bn = soft(yW - gW / L, P.lam / L), yb - gb / L
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        yW = Wn + ((t - 1) / tn) * (Wn - W)
        yb = bn + ((t - 1) / tn) * (bn - b)
        W, b, t = Wn, bn, tn
        hist.append(P.F(W, b))
    return W, b, np.array(hist)


def fista_backtracking(P, L0, iters, eta=2.0):
    """FISTA with Beck & Teboulle backtracking: increase L until the quadratic
    upper model f(y) + <g, x-y> + L/2 ||x-y||^2 really upper-bounds f(x)."""
    W, b = np.zeros((P.C, P.p)), np.zeros(P.C)
    yW, yb, t, L = W.copy(), b.copy(), 1.0, L0
    hist, Ls = [], []
    for _ in range(iters):
        fy = P.f(yW, yb)
        gW, gb = P.grad(yW, yb)
        while True:
            Wn, bn = soft(yW - gW / L, P.lam / L), yb - gb / L
            dW, db = Wn - yW, bn - yb
            model = fy + np.sum(gW * dW) + np.sum(gb * db) + L / 2 * (np.sum(dW ** 2) + np.sum(db ** 2))
            if P.f(Wn, bn) <= model + 1e-12:
                break
            L *= eta
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        yW = Wn + ((t - 1) / tn) * (Wn - W)
        yb = bn + ((t - 1) / tn) * (bn - b)
        W, b, t = Wn, bn, tn
        hist.append(P.F(W, b))
        Ls.append(L)
    return W, b, np.array(hist), np.array(Ls)


def solve_cvxpy(P):
    W = cp.Variable((P.C, P.p))
    b = cp.Variable(P.C)
    Z = P.X @ W.T + np.ones((P.n, 1)) @ cp.reshape(b, (1, P.C), order="C")
    kl = P.qlogq + cp.sum(cp.log_sum_exp(Z / P.T, axis=1) - cp.sum(cp.multiply(P.Q, Z / P.T), axis=1)) / P.n
    ce = cp.sum(cp.log_sum_exp(Z, axis=1) - cp.sum(cp.multiply(P.Y, Z), axis=1)) / P.n
    obj = P.alpha * P.T ** 2 * kl + (1 - P.alpha) * ce + P.lam * cp.norm1(W)
    prob = cp.Problem(cp.Minimize(obj))
    t0 = time.time()
    prob.solve(solver=cp.CLARABEL)
    return W.value, b.value, prob.value, time.time() - t0, prob.status


def lambda_max(P, L):
    """Smallest lam for which W = 0 is optimal: lam_max = ||grad_W f(0, b*)||_inf,
    with b* the best bias when W = 0 (found by gradient descent on b alone)."""
    W, b = np.zeros((P.C, P.p)), np.zeros(P.C)
    for _ in range(3000):
        b = b - P.grad(W, b)[1] / 0.5          # f restricted to b is (1/2)-smooth
    return np.abs(P.grad(W, b)[0]).max()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion")
    # n is kept small because cvxpy's interior-point solve grows quickly with n
    # (n = 300 already takes ~1 min); our first-order methods do not need this.
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--lam-frac", type=float, default=0.05, help="lam = lam_frac * lam_max")
    ap.add_argument("--out", default="results/convex")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- data: n training images, 2x2 average-pooled to 14x14 = 196 features
    data = load_dataset(args.dataset, device="cpu")
    x, y = data["train"][0][:args.n], data["train"][1][:args.n]
    teacher = TeacherCNN()
    teacher.load_state_dict(torch.load(f"results/teacher_{args.dataset}.pt", map_location="cpu"))
    teacher.eval()
    with torch.no_grad():
        tlog = teacher(x).double().numpy()
    X = tF.avg_pool2d(x, 2).flatten(1).double().numpy()
    Y = np.eye(10)[y.numpy()]
    Q = softmax(tlog / args.T)

    P = KDProblem(X, Y, Q, args.T, args.alpha, lam=0.0)
    L = P.lipschitz()
    lmax = lambda_max(P, L)
    P.lam = args.lam_frac * lmax
    print(f"n={P.n} p={P.p} L={L:.4f} lam_max={lmax:.5f} lam={P.lam:.6f}")

    # ---- gradient check (central finite differences along random directions)
    W0, b0 = rng.normal(size=(P.C, P.p)) * 0.01, rng.normal(size=P.C) * 0.01
    gW, gb = P.grad(W0, b0)
    errs = []
    for _ in range(5):
        dW, db = rng.normal(size=W0.shape), rng.normal(size=b0.shape)
        eps = 1e-6
        fd = (P.f(W0 + eps * dW, b0 + eps * db) - P.f(W0 - eps * dW, b0 - eps * db)) / (2 * eps)
        an = np.sum(gW * dW) + np.sum(gb * db)
        errs.append(abs(fd - an) / max(1e-12, abs(an)))
    print(f"gradient check: max relative error {max(errs):.2e}")

    # ---- our solvers
    t0 = time.time(); Wi, bi, hi = ista(P, L, args.iters); t_ista = time.time() - t0
    t0 = time.time(); Wf, bf, hf = fista(P, L, args.iters); t_fista = time.time() - t0
    t0 = time.time(); Wb, bb, hb, Ls = fista_backtracking(P, L / 50, args.iters); t_bt = time.time() - t0

    print(f"our solvers done: ISTA {t_ista:.1f}s, FISTA {t_fista:.1f}s, FISTA-BT {t_bt:.1f}s; solving with cvxpy...")

    # ---- reference solution
    Wc, bc, _, t_cvx, status = solve_cvxpy(P)
    Fc = P.F(Wc, bc)
    Fstar = min(Fc, hi.min(), hf.min(), hb.min())
    print(f"cvxpy: status={status} F={Fc:.10f} time={t_cvx:.1f}s")
    print(f"ISTA  F={hi[-1]:.10f}  FISTA F={hf[-1]:.10f}  FISTA-BT F={hb[-1]:.10f}")

    R2 = np.sum(Wc ** 2) + np.sum(bc ** 2)              # ||theta_0 - theta*||^2 with theta_0 = 0
    supp_cvx = np.abs(Wc) > 1e-6 * np.abs(Wc).max()     # interior point gives near-zeros, not exact zeros
    supp_f = Wf != 0
    res = {
        "n": P.n, "p": P.p, "T": args.T, "alpha": args.alpha,
        "L": L, "lam_max": lmax, "lam": P.lam,
        "grad_check_max_rel_err": max(errs),
        "cvxpy": {"status": status, "F": Fc, "seconds": t_cvx, "nnz_W": int(supp_cvx.sum())},
        "ista": {"F": hi[-1], "gap": hi[-1] - Fstar, "seconds": t_ista, "nnz_W": int((Wi != 0).sum())},
        "fista": {"F": hf[-1], "gap": hf[-1] - Fstar, "seconds": t_fista, "nnz_W": int(supp_f.sum())},
        "fista_bt": {"F": hb[-1], "gap": hb[-1] - Fstar, "seconds": t_bt, "final_L": Ls[-1]},
        "W_rel_err_fista_vs_cvx": float(np.linalg.norm(Wf - Wc) / np.linalg.norm(Wc)),
        "support_jaccard_fista_vs_cvx": float((supp_cvx & supp_f).sum() / (supp_cvx | supp_f).sum()),
        "W_total": P.C * P.p,
        "iters_to_1e-6_gap": {
            k: (int(np.argmax(h - Fstar < 1e-6)) + 1 if (h - Fstar < 1e-6).any() else None)
            for k, h in [("ista", hi), ("fista", hf), ("fista_bt", hb)]},
    }
    with open(os.path.join(args.out, "convex_check.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))

    # ---- plot 1: convergence vs theory
    k = np.arange(1, args.iters + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    floor = 1e-13
    ax.loglog(k, np.maximum(hi - Fstar, floor), label="ISTA (step 1/L)")
    ax.loglog(k, np.maximum(hf - Fstar, floor), label="FISTA (step 1/L)")
    ax.loglog(k, np.maximum(hb - Fstar, floor), label="FISTA + backtracking")
    ax.loglog(k, L * R2 / (2 * k), "k--", lw=1, label=r"ISTA bound $L R^2/(2k)$")
    ax.loglog(k, 2 * L * R2 / (k + 1) ** 2, "k:", lw=1, label=r"FISTA bound $2LR^2/(k+1)^2$")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$F(\theta_k) - F^\star$")
    ax.set_ylim(bottom=1e-12)
    ax.set_title("Convex KD (linear student): convergence vs theory")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "convergence.png"), dpi=160)

    # ---- plot 2: the sparse linear student's weights, one 14x14 image per class
    names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"] \
        if args.dataset == "fashion" else [str(i) for i in range(10)]
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    vmax = np.abs(Wf).max()
    for c, a in enumerate(axes.flat):
        a.imshow(Wf[c].reshape(14, 14), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(f"{names[c]} ({int((Wf[c] != 0).sum())} nz)", fontsize=8)
        a.axis("off")
    fig.suptitle(f"FISTA solution, lam = {args.lam_frac} lam_max: "
                 f"{int(supp_f.sum())}/{P.C * P.p} non-zero weights (white = 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "weights.png"), dpi=160)


if __name__ == "__main__":
    main()
