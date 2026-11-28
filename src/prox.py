# After a gradient step v, each operator minimizes
# h(w) + ||w-v||^2/(2*eta) for its own sparsity penalty h.
import torch


def soft_threshold(v: torch.Tensor, tau: float) -> torch.Tensor:
    # Return the L1 prox: values within +/-tau become zero; others shrink.
    return v.sign() * (v.abs() - tau).clamp_min(0.0)


def topk_masks(tensors, k: int):
    # Return one 0/1 mask per tensor, keeping the global k largest magnitudes.
    # This projects the concatenated weights onto the L0 budget for IHT.
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
    # Modify W and b in place. One group is a weight row plus its bias;
    # scaling by max(1-tau/||group||, 0) can remove the entire neuron.
    rows = torch.cat([W.reshape(W.shape[0], -1), b.unsqueeze(1)], dim=1)
    norms = rows.norm(dim=1)
    scale = (1.0 - tau / norms.clamp_min(1e-12)).clamp_min(0.0)
    W.mul_(scale.view(-1, *([1] * (W.dim() - 1))))
    b.mul_(scale)
