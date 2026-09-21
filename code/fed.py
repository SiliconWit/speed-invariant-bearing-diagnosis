"""Federated training: FedAvg, FedProx and prototype alignment.

Clients are (speed, shard) pairs, so speed heterogeneity between clients is the
only systematic difference between their data distributions.

Prototype alignment sends C x k floats per client per round, 256 numbers here,
against a second network for adversarial alignment or a full model copy for
FedProx's anchor.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from models import Net

N_CLASS = 4


def make_clients(X, y, s, speeds, shards=3, seed=0):
    """Split the training speeds into shards; each (speed, shard) is one client."""
    rng = np.random.default_rng(seed)
    clients = []
    for sp in speeds:
        idx = np.where(s == sp)[0]
        rng.shuffle(idx)
        for part in np.array_split(idx, shards):
            clients.append(dict(idx=part, speed=sp))
    return clients


def _loader(X, y, idx, bs, rng):
    order = rng.permutation(len(idx))
    for i in range(0, len(order), bs):
        sel = idx[order[i:i + bs]]
        yield torch.from_numpy(X[sel]), torch.from_numpy(y[sel])


def local_update(model, X, y, idx, epochs, lr, bs, rng, method,
                 global_proto=None, lam=1.0, mu=0.01, global_state=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in _loader(X, y, idx, bs, rng):
            opt.zero_grad()
            logits, z = model(xb)
            loss = F.cross_entropy(logits, yb)
            if method == "proto" and global_proto is not None:
                p = global_proto[yb]
                valid = ~torch.isnan(p[:, 0])
                if valid.any():
                    # pull each sample toward its class's global prototype
                    loss = loss + lam * ((z[valid] - p[valid]) ** 2).sum(1).mean()
            if method == "prox" and global_state is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_state))
                loss = loss + 0.5 * mu * prox
            loss.backward()
            opt.step()
    return model


def client_prototypes(model, X, y, idx):
    """Mean embedding per class, renormalised to the sphere. NaN where absent."""
    model.eval()
    with torch.no_grad():
        _, z = model(torch.from_numpy(X[idx]))
    z = z.numpy()
    yy = y[idx]
    P = np.full((N_CLASS, z.shape[1]), np.nan, dtype=np.float32)
    for c in range(N_CLASS):
        m = yy == c
        if m.sum() > 0:
            v = z[m].mean(0)
            P[c] = v / (np.linalg.norm(v) + 1e-12)
    return P, np.array([(yy == c).sum() for c in range(N_CLASS)])


def average_states(states, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            out[k] = sum(float(wi) * st[k] for wi, st in zip(w, states))
        else:
            out[k] = states[0][k]
    return out


def train_federated(X, y, s, train_speeds, method="fedavg", rounds=60,
                    local_epochs=2, lr=1e-3, bs=64, shards=3, lam=1.0, mu=0.01,
                    seed=0, nbins=256, k=64, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    clients = make_clients(X, y, s, train_speeds, shards, seed)
    glob = Net(nbins, k, N_CLASS)
    global_proto = None

    for rnd in range(rounds):
        states, protos, counts, sizes = [], [], [], []
        gstate = [p.detach().clone() for p in glob.parameters()]
        gp = None if global_proto is None else torch.from_numpy(global_proto)
        for cl in clients:
            m = copy.deepcopy(glob)
            local_update(m, X, y, cl["idx"], local_epochs, lr, bs, rng, method,
                         global_proto=gp, lam=lam, mu=mu, global_state=gstate)
            states.append(m.state_dict())
            sizes.append(len(cl["idx"]))
            if method == "proto":
                P, n = client_prototypes(m, X, y, cl["idx"])
                protos.append(P)
                counts.append(n)
        glob.load_state_dict(average_states(states, sizes))

        if method == "proto":
            P = np.stack(protos)                    # (clients, C, k)
            N = np.stack(counts).astype(float)      # (clients, C)
            agg = np.full((N_CLASS, P.shape[2]), np.nan, dtype=np.float32)
            for c in range(N_CLASS):
                ok = ~np.isnan(P[:, c, 0]) & (N[:, c] > 0)
                if ok.any():
                    w = N[ok, c] / N[ok, c].sum()
                    v = (w[:, None] * P[ok, c]).sum(0)
                    agg[c] = v / (np.linalg.norm(v) + 1e-12)
            global_proto = agg
        if log is not None and (rnd + 1) % 10 == 0:
            log(rnd + 1, glob, global_proto)
    return glob, global_proto


@torch.no_grad()
def embed(model, X, bs=512):
    model.eval()
    zs, ls = [], []
    for i in range(0, len(X), bs):
        logit, z = model(torch.from_numpy(X[i:i + bs]))
        zs.append(z.numpy()); ls.append(logit.numpy())
    return np.concatenate(zs), np.concatenate(ls)


def accuracy(model, X, y, proto=None):
    z, logits = embed(model, X)
    lin = float((logits.argmax(1) == y).mean())
    npx = np.nan
    if proto is not None and not np.isnan(proto).all():
        d = ((z[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
        npx = float((np.nanargmin(d, axis=1) == y).mean())
    return lin, npx, z
