"""Centralised training, to separate the representation question from federation.

Separates the effect of the representation from the effect of federation.
"""
import numpy as np, torch, torch.nn.functional as F
from models import Net


def train_central(X, y, sel, rounds=60, lr=1e-3, bs=64, seed=0, balance=True,
                  nbins=256, k=64, n_class=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.where(sel)[0]
    w = None
    if balance:
        cnt = np.bincount(y[idx], minlength=n_class).astype(float)
        w = torch.tensor((cnt.sum() / (n_class * np.maximum(cnt, 1))), dtype=torch.float32)
    m = Net(nbins, k, n_class)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(rounds):
        m.train()
        order = rng.permutation(len(idx))
        for i in range(0, len(order), bs):
            sub = idx[order[i:i + bs]]
            opt.zero_grad()
            logits, z = m(torch.from_numpy(X[sub]))
            F.cross_entropy(logits, torch.from_numpy(y[sub]), weight=w).backward()
            opt.step()
    return m


@torch.no_grad()
def acc(m, X, y):
    m.eval()
    logits, _ = m(torch.from_numpy(X))
    return float((logits.argmax(1).numpy() == y).mean())
