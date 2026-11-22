"""Measuring model size: unstructured (non-zeros) and structured (neurons)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def prunable_weights(model):
    """The tensors the budget applies to: weights of Linear/Conv layers.
    Biases are excluded (410 of 266k numbers in the student) but counted in size."""
    return [m.weight for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]


@torch.no_grad()
def size_report(model):
    ws = prunable_weights(model)
    n_total = sum(p.numel() for p in model.parameters())
    nnz_total = sum(int((p != 0).sum()) for p in model.parameters())
    d = sum(w.numel() for w in ws)
    nnz_w = sum(int((w != 0).sum()) for w in ws)
    return {
        "params_total": n_total,
        "nnz_total": nnz_total,
        "prunable": d,
        "nnz_weights": nnz_w,
        "density": nnz_w / d,
        # storage: dense = 4 bytes/param; sparse = (4-byte value + 4-byte index)
        # per non-zero weight + dense biases
        "dense_kb": n_total * 4 / 1024,
        "sparse_kb": (nnz_w * 8 + (n_total - d) * 4) / 1024,
        "layer_nnz": [int((w != 0).sum()) for w in ws],
        "layer_size": [w.numel() for w in ws],
    }


@torch.no_grad()
def compact_mlp(model):
    """Physically remove dead inputs / neurons from a StudentMLP.

    * an input pixel whose column in W1 is all-zero is never used -> drop it
    * a hidden neuron with an all-zero incoming row outputs the constant
      relu(b_j); fold that constant into the next layer's bias and drop it
    * a hidden neuron whose outgoing column is all-zero has no effect -> drop it

    Returns (input_index, [W], [b]) describing an equivalent *smaller dense*
    network. Its parameter count is the "structured size": what you would
    actually ship and run without any sparse-matrix support.
    """
    Ws = [l.weight.detach().clone() for l in model.layers]
    bs = [l.bias.detach().clone() for l in model.layers]
    in_idx = (Ws[0] != 0).any(dim=0).nonzero().flatten()
    Ws[0] = Ws[0][:, in_idx]
    for l in range(len(Ws) - 1):
        row_alive = (Ws[l] != 0).any(dim=1)
        const = F.relu(bs[l]) * (~row_alive)          # constant outputs of dead-input neurons
        bs[l + 1] = bs[l + 1] + Ws[l + 1] @ const
        col_alive = (Ws[l + 1] != 0).any(dim=0)
        keep = row_alive & col_alive
        Ws[l], bs[l] = Ws[l][keep], bs[l][keep]
        Ws[l + 1] = Ws[l + 1][:, keep]
    return in_idx, Ws, bs


def compact_forward(in_idx, Ws, bs, x):
    h = x.flatten(1)[:, in_idx]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.t() + b
        if i < len(Ws) - 1:
            h = F.relu(h)
    return h


@torch.no_grad()
def structured_report(model, x_check=None):
    """Neurons alive per layer and dense parameter count of the compacted net.
    If x_check is given, also verifies the compact net gives the same logits."""
    in_idx, Ws, bs = compact_mlp(model)
    out = {
        "inputs_alive": int(in_idx.numel()),
        "neurons_alive": [int(W.shape[0]) for W in Ws[:-1]],
        "compact_params": int(sum(W.numel() + b.numel() for W, b in zip(Ws, bs))),
    }
    if x_check is not None:
        was_training = model.training
        model.eval()
        diff = (model(x_check) - compact_forward(in_idx, Ws, bs, x_check)).abs().max()
        model.train(was_training)
        out["compact_max_abs_diff"] = float(diff)
    return out
