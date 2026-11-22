"""Dataset loading. Everything is kept as tensors (on the GPU if available),
which is much faster than a DataLoader for 28x28 images."""
import torch
import torch.nn.functional as F
from torchvision import datasets

# per-dataset pixel mean / std, used to standardise inputs
STATS = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}
CLASSES = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}


def load_dataset(name="fashion", root="data", device="cpu", val_size=5000, split_seed=0):
    """Returns {"train": (x, y), "val": (x, y), "test": (x, y)}.

    The official 60k training set is split into 55k train / 5k validation with
    a fixed seed (so every script sees the same split). Validation is used to
    choose hyper-parameters (T, alpha); the test set is only used for reporting.
    """
    cls = CLASSES[name]
    mean, std = STATS[name]

    def to_xy(ds):
        x = (ds.data.float() / 255.0 - mean) / std
        return x.unsqueeze(1), ds.targets.clone().long()

    xtr, ytr = to_xy(cls(root, train=True, download=True))
    xte, yte = to_xy(cls(root, train=False, download=True))
    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(split_seed))
    va, tr = perm[:val_size], perm[val_size:]
    put = lambda x, y: (x.to(device), y.to(device))
    return {"train": put(xtr[tr], ytr[tr]), "val": put(xtr[va], ytr[va]), "test": put(xte, yte)}


def batches(tensors, batch_size, shuffle=True):
    """Yield aligned mini-batches from a tuple of tensors with equal first dim."""
    n = tensors[0].shape[0]
    dev = tensors[0].device
    idx = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        yield tuple(t[b] for t in tensors)


def augment(x, pad=2, flip=True):
    """Random translation (pad + random crop, per sample) and optional
    horizontal flip. Used only for training the teacher."""
    n, _, h, w = x.shape
    if flip:
        m = torch.rand(n, device=x.device) < 0.5
        x = torch.where(m.view(n, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad), value=float(x.min()))[:, 0]
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i.view(n, 1) + torch.arange(h, device=x.device)).view(n, h, 1)
    cols = (j.view(n, 1) + torch.arange(w, device=x.device)).view(n, 1, w)
    return xp[torch.arange(n, device=x.device).view(n, 1, 1), rows, cols].unsqueeze(1)
