"""Stochastic (accelerated) proximal gradient training.

Problem:   min_w  F(w) + h(w)
   F = distillation loss (smooth, estimated on mini-batches)
   h = the non-smooth "size" term (l1, group lasso, l0-ball indicator, mask)

Iteration (with extrapolation / momentum beta, FISTA-style):
   y_k     = w_k + beta * (w_k - w_{k-1})
   w_{k+1} = prox_{eta_k * h}( y_k - eta_k * grad F_batch(y_k) )

beta = 0 gives plain proximal SGD (stochastic ISTA). The gradient comes from
PyTorch autograd; the update rule, the extrapolation, the step-size schedule
and all proximal operators are implemented here (no torch.optim is used).
"""
import math
import torch

from .prox import soft_threshold, topk_masks, block_soft_threshold_rows
from .sparsity import prunable_weights, size_report
from .data import batches


# ----------------------------------------------------------------------------
# Non-smooth terms h(w). Each one knows its value (for logging the objective)
# and its proximal operator (applied in place after every gradient step).
# ----------------------------------------------------------------------------
class NoReg:
    name = "none"

    def penalty(self, model):
        return 0.0

    def prox(self, model, eta, it):
        pass


class L1(NoReg):
    """h(w) = lam * ||w||_1  over all prunable weights."""
    name = "l1"

    def __init__(self, lam):
        self.lam = lam

    @torch.no_grad()
    def penalty(self, model):
        return self.lam * sum(float(w.abs().sum()) for w in prunable_weights(model))

    @torch.no_grad()
    def prox(self, model, eta, it):
        for w in prunable_weights(model):
            w.copy_(soft_threshold(w, eta * self.lam))


class GroupLasso(NoReg):
    """h(w) = lam * sum_g sqrt(|g|) * ||w_g||_2, groups = neurons of hidden layers.

    Group g = incoming weights + bias of one hidden neuron. The sqrt(|g|)
    weight is the standard group-lasso normalisation so that layers with
    longer rows (784 inputs vs 300) are not penalised unfairly.
    The output layer is not grouped (we cannot delete a class).
    """
    name = "group"

    def __init__(self, lam):
        self.lam = lam

    def _hidden(self, model):
        return list(model.layers)[:-1]

    @torch.no_grad()
    def penalty(self, model):
        total = 0.0
        for layer in self._hidden(model):
            rows = torch.cat([layer.weight, layer.bias.unsqueeze(1)], 1)
            total += math.sqrt(rows.shape[1]) * float(rows.norm(dim=1).sum())
        return self.lam * total

    @torch.no_grad()
    def prox(self, model, eta, it):
        for layer in self._hidden(model):
            tau = eta * self.lam * math.sqrt(layer.weight.shape[1] + 1)
            block_soft_threshold_rows(layer.weight, layer.bias, tau)


class HardBudget(NoReg):
    """h(w) = indicator{ ||w||_0 <= k }  ->  prox = keep top-k magnitudes (IHT).

    Starting a randomly initialised network at its final tiny budget throws
    away information before the weights mean anything, so the budget is
    tightened gradually: k_t goes from d (dense) to k_final over the first
    `ramp` fraction of iterations with the cubic schedule of Zhu & Gupta (2017),
    then stays at k_final. ramp = 0 gives IHT with a fixed budget from step 0.
    Every iterate after the ramp satisfies the constraint exactly.
    """
    name = "iht"

    def __init__(self, k_final, total_steps, ramp=0.5):
        self.k_final, self.total_steps, self.ramp = int(k_final), total_steps, ramp
        self.d = None

    def k_at(self, it):
        ramp_steps = int(self.ramp * self.total_steps)
        if it >= ramp_steps:
            return self.k_final
        frac = 1.0 - it / ramp_steps
        return int(self.k_final + (self.d - self.k_final) * frac ** 3)

    @torch.no_grad()
    def prox(self, model, eta, it):
        ws = prunable_weights(model)
        if self.d is None:
            self.d = sum(w.numel() for w in ws)
        for w, m in zip(ws, topk_masks(ws, self.k_at(it))):
            w.mul_(m)


class FixedMask(NoReg):
    """h(w) = indicator{ w is zero outside a fixed support }  ->  prox = w * mask.

    Used to fine-tune ("debias") a sparse model: the support found by l1 /
    IHT / pruning is frozen and the surviving weights are re-optimised without
    any shrinkage penalty.
    """
    name = "mask"

    def __init__(self, masks):
        self.masks = masks

    @torch.no_grad()
    def prox(self, model, eta, it):
        for w, m in zip(prunable_weights(model), self.masks):
            w.mul_(m)


def support_masks(model):
    return [(w != 0).float() for w in prunable_weights(model)]


# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, batch_size=2000):
    model.eval()
    correct = 0
    for xb, yb in batches((x, y), batch_size, shuffle=False):
        correct += int((model(xb).argmax(1) == yb).sum())
    model.train()
    return correct / len(y)


def train_prox(model, data, loss_fn, reg=None, epochs=20, lr=0.05, beta=0.9,
               batch_size=128, teacher_logits=None, log=print, tag="", warmup=1.0,
               backtrack=True, max_backtracks=20):
    """Stochastic accelerated proximal gradient on F + h.

    data           : dict from load_dataset (uses "train" and "val")
    loss_fn        : f(student_logits, teacher_logits, y) -> scalar F on the batch
    reg            : one of the h(w) classes above (NoReg if None)
    lr             : peak step size; linear warm-up for `warmup` epochs, then
                     cosine decay to 0 (large early steps + momentum + strong
                     shrinkage made big-lambda group-lasso runs diverge)
    beta           : extrapolation weight (0 = plain proximal SGD / ISTA)
    teacher_logits : precomputed teacher logits aligned with data["train"]
    backtrack      : if True, the scheduled eta is only an upper bound; each
                     step is shrunk by halving until the sufficient-decrease
                     condition holds on the mini-batch (prevents blow-ups once
                     the network is very sparse and the curvature is large)

    Returns a history list with one dict per epoch.
    """
    reg = reg or NoReg()
    xtr, ytr = data["train"]
    tl = teacher_logits if teacher_logits is not None else torch.zeros(len(ytr), 1, device=xtr.device)
    params = list(model.parameters())
    w_prev = [p.detach().clone() for p in params]
    steps_per_epoch = math.ceil(len(ytr) / batch_size)
    total = epochs * steps_per_epoch
    if isinstance(reg, HardBudget):
        reg.total_steps = total
    history, it = [], 0
    model.train()
    for ep in range(epochs):
        run_loss, nb, n_backtracks = 0.0, 0, 0
        for xb, yb, tb in batches((xtr, ytr, tl), batch_size):
            # cosine-decayed step size with a linear warm-up over `warmup` epochs
            eta = lr * 0.5 * (1.0 + math.cos(math.pi * it / total))
            eta *= min(1.0, (it + 1) / max(1, warmup * steps_per_epoch))
            with torch.no_grad():                       # extrapolation y_k
                for p, q in zip(params, w_prev):
                    diff = p - q
                    q.copy_(p)
                    p.add_(diff, alpha=beta)
            loss = loss_fn(model(xb), tb if teacher_logits is not None else None, yb)
            for p in params:
                p.grad = None
            loss.backward()                             # grad of F at y_k
            with torch.no_grad():
                if not backtrack:
                    for p in params:                    # forward (gradient) step
                        p.add_(p.grad, alpha=-eta)
                    reg.prox(model, eta, it)            # backward (proximal) step
                else:
                    # Backtracking line search on the current mini-batch: accept the
                    # prox-gradient step w+ only if the quadratic upper model holds,
                    #   F(w+) <= F(y) + <g, w+ - y> + ||w+ - y||^2 / (2 eta),
                    # i.e. eta is small enough for the *local* curvature. Otherwise
                    # halve eta and retry (same test as in convex_check.py).
                    y = [p.detach().clone() for p in params]
                    g = [p.grad for p in params]
                    Fy = loss.item()
                    for _ in range(max_backtracks):
                        for p, yp, gp in zip(params, y, g):
                            p.copy_(yp).add_(gp, alpha=-eta)
                        reg.prox(model, eta, it)
                        Fn = loss_fn(model(xb), tb if teacher_logits is not None else None, yb).item()
                        lin, sq = 0.0, 0.0
                        for p, yp, gp in zip(params, y, g):
                            d = p - yp
                            lin += float((gp * d).sum())
                            sq += float((d * d).sum())
                        if Fn <= Fy + lin + sq / (2 * eta) + 1e-7:
                            break
                        eta *= 0.5
                        n_backtracks += 1
            run_loss += loss.item()
            nb += 1
            it += 1
            if not math.isfinite(run_loss):
                log(f"{tag} DIVERGED at iteration {it} (loss is not finite) - stopping this run")
                history.append({"epoch": ep + 1, "diverged": True})
                return history
        sz = size_report(model)
        rec = {
            "epoch": ep + 1,
            "F": run_loss / nb,
            "h": reg.penalty(model),
            "val_acc": evaluate(model, *data["val"]),
            "nnz_weights": sz["nnz_weights"],
            "density": sz["density"],
            "eta": eta,
            "backtracks_per_step": n_backtracks / nb,
        }
        rec["objective"] = rec["F"] + rec["h"]
        history.append(rec)
        log(f"{tag} ep {ep + 1:3d}  F {rec['F']:.4f}  h {rec['h']:.4f}  "
            f"val {rec['val_acc']:.4f}  density {rec['density']:.4f}  "
            f"backtracks/step {rec['backtracks_per_step']:.2f}")
    return history
