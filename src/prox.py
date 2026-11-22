"""Proximal operators used by the solvers.

Every sparse method in this project is "proximal gradient descent" with a
different non-smooth term h(w). The only thing that changes between methods is
the proximal operator

    prox_{eta*h}(v) = argmin_w  h(w) + (1 / (2*eta)) * ||w - v||^2 ,

which is implemented here in closed form for each h we use.
"""
import torch


def soft_threshold(v: torch.Tensor, tau: float) -> torch.Tensor:
    """prox of tau*||.||_1  ->  sign(v) * max(|v| - tau, 0)   (element-wise).

    Entries with |v| <= tau become exactly zero; the rest shrink towards zero
    by tau. This is what makes the l1 method produce exact zeros.
    """
    return v.sign() * (v.abs() - tau).clamp_min(0.0)


def topk_masks(tensors, k: int):
    """Euclidean projection onto the l0 ball {w : ||w||_0 <= k}.

    The projection keeps the k entries of largest magnitude (taken jointly over
    *all* tensors, i.e. a global budget) and zeroes the rest. We return one 0/1
    mask per tensor; the caller multiplies the weights by them.

    This is the prox of the indicator function of the l0 ball, so using it
    inside proximal gradient gives Iterative Hard Thresholding (IHT).
    """
    flat = torch.cat([t.reshape(-1) for t in tensors])
    d = flat.numel()
    if k >= d:
        return [torch.ones_like(t) for t in tensors]
    keep = torch.zeros(d, device=flat.device, dtype=flat.dtype)
    if k > 0:
        # topk on indices (not a threshold) so ties can never keep > k entries
        keep[torch.topk(flat.abs(), k, sorted=False).indices] = 1.0
    masks, start = [], 0
    for t in tensors:
        n = t.numel()
        masks.append(keep[start:start + n].view_as(t))
        start += n
    return masks


def block_soft_threshold_rows(W: torch.Tensor, b: torch.Tensor, tau: torch.Tensor):
    """prox of sum_g tau_g * ||w_g||_2 where group g = (row g of W, b[g]).

    One row of a Linear weight matrix plus its bias are all the *incoming*
    parameters of one neuron. The prox scales every group by
    max(1 - tau_g / ||w_g||, 0): a group with small norm is set to zero as a
    whole, which deletes the neuron. Works in place. `tau` may be a scalar or
    a vector with one threshold per row.
    """
    rows = torch.cat([W.reshape(W.shape[0], -1), b.unsqueeze(1)], dim=1)
    norms = rows.norm(dim=1)
    scale = (1.0 - tau / norms.clamp_min(1e-12)).clamp_min(0.0)
    W.mul_(scale.view(-1, *([1] * (W.dim() - 1))))
    b.mul_(scale)
