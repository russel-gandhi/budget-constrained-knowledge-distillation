"""Checks that each closed-form proximal operator really is
argmin_w h(w) + 1/(2*eta)||w - v||^2, by comparing it with random perturbations,
plus a check that the compacted student network is exactly equivalent.

    python -m tests.test_prox
"""
import torch

from src.prox import soft_threshold, topk_masks, block_soft_threshold_rows
from src.models import StudentMLP
from src.solvers import L1, GroupLasso, HardBudget
from src.sparsity import structured_report, prunable_weights

torch.manual_seed(0)


def prox_objective(h, w, v, eta):
    return h(w) + ((w - v) ** 2).sum() / (2 * eta)


def check_is_minimiser(h, prox_v, v, eta, trials=2000, scale=1e-2):
    best = prox_objective(h, prox_v, v, eta)
    for _ in range(trials):
        cand = prox_v + scale * torch.randn_like(prox_v)
        assert prox_objective(h, cand, v, eta) >= best - 1e-9, "found a better point than the prox!"


def test_soft_threshold():
    v, eta, lam = torch.randn(50, dtype=torch.float64), 0.3, 1.0
    p = soft_threshold(v, eta * lam)
    check_is_minimiser(lambda w: lam * w.abs().sum(), p, v, eta)
    assert ((v.abs() <= eta * lam) == (p == 0)).all()


def test_block_soft_threshold():
    W, b = torch.randn(6, 5, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    v = torch.cat([W, b[:, None]], 1).clone()
    tau = 2.0
    block_soft_threshold_rows(W, b, tau)
    p = torch.cat([W, b[:, None]], 1)
    check_is_minimiser(lambda w: tau * w.norm(dim=1).sum(), p, v, 1.0)


def test_topk_projection():
    ts = [torch.randn(30, 20), torch.randn(10, 7)]
    k = 37
    ms = topk_masks(ts, k)
    assert int(sum(m.sum() for m in ms)) == k
    kept = torch.cat([(t * m).flatten() for t, m in zip(ts, ms)])
    allv = torch.cat([t.flatten() for t in ts])
    # projection onto the l0 ball keeps the k largest |v|: nothing dropped is bigger than anything kept
    dropped = allv[kept == 0].abs()
    assert dropped.max() <= kept[kept != 0].abs().min()


def test_compact_equivalence():
    m = StudentMLP().double()
    with torch.no_grad():
        GroupLasso(1.0).prox(m, 0.02, 0)              # kill many neurons
        HardBudget(20000, 1, 0.0).prox(m, 0.0, 0)
        m.layers[0].weight[:, :100] = 0               # unused input pixels
    rep = structured_report(m, torch.randn(64, 1, 28, 28, dtype=torch.float64))
    assert rep["compact_max_abs_diff"] < 1e-10, rep
    assert rep["compact_params"] < sum(p.numel() for p in m.parameters())


def test_l1_reg_matches_soft_threshold():
    m = StudentMLP()
    w0 = [w.detach().clone() for w in prunable_weights(m)]
    L1(0.5).prox(m, 0.1, 0)
    for w, v in zip(prunable_weights(m), w0):
        assert torch.allclose(w, soft_threshold(v, 0.05))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
