"""The margin ratio: prototype geometry against speed-induced displacement.

Three measurable quantities.  Delta is how far apart the class prototypes are,
r is how spread out a class is around its own prototype, and delta is how far a
speed change moves an embedding.  Proposition 1 says a nearest-prototype decision
survives the speed change whenever r + delta < Delta/2, so the ratio

    M = Delta / (2 (r + delta))

should cross 1 at roughly the point where accuracy starts to fall.  The claim is
deliberately narrow: it is a deterministic statement about a nearest-prototype
rule, not a generalisation bound, and it says nothing about samples already
outside their radius before the speed change.
"""
import numpy as np

N_CLASS = 4


def prototypes(z, y, n_class=N_CLASS):
    P = np.full((n_class, z.shape[1]), np.nan, dtype=float)
    for c in range(n_class):
        m = y == c
        if m.sum():
            v = z[m].mean(0)
            P[c] = v / (np.linalg.norm(v) + 1e-12)
    return P


def radii(z, y, P, q=95.0, n_class=N_CLASS):
    r = np.full(n_class, np.nan)
    for c in range(n_class):
        m = y == c
        if m.sum():
            r[c] = np.percentile(np.linalg.norm(z[m] - P[c], axis=1), q)
    return r


def separation(P):
    ok = ~np.isnan(P[:, 0])
    Q = P[ok]
    d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def displacement_paired(z_a, z_b):
    """Simulation only: the same realisation seen at two speeds, so the shift
    of an individual sample is directly observable."""
    return float(np.median(np.linalg.norm(z_a - z_b, axis=1)))


def displacement_means(z_a, y_a, z_b, y_b, n_class=N_CLASS):
    """Real data: no pairing exists, so use the shift of the class means.

    In simulation it both over- and underestimates the per-sample displacement,
    so it is not a bound in either direction.  It is the only estimate available
    without a synchronised second recording of the same defect.
    """
    Pa, Pb = prototypes(z_a, y_a, n_class), prototypes(z_b, y_b, n_class)
    d = [np.linalg.norm(Pa[c] - Pb[c]) for c in range(n_class)
         if not (np.isnan(Pa[c, 0]) or np.isnan(Pb[c, 0]))]
    return float(np.mean(d)), float(np.max(d))


def margin_ratio(Delta, r_bar, delta):
    return float(Delta / (2.0 * (r_bar + delta)))


def geometry(z_tr, y_tr, z_te, y_te, q=95.0):
    """Everything the proposition needs, from a training and a held-out set."""
    P = prototypes(z_tr, y_tr)
    r = radii(z_tr, y_tr, P, q)
    Delta = separation(P)
    d_mean, d_max = displacement_means(z_tr, y_tr, z_te, y_te)
    r_bar = float(np.nanmax(r))
    return dict(Delta=Delta, r=r.tolist(), r_bar=r_bar,
                delta_mean=d_mean, delta_max=d_max,
                M=margin_ratio(Delta, r_bar, d_mean),
                M_max=margin_ratio(Delta, r_bar, d_max),
                proto=P)


def nearest_prototype_acc(z, y, P):
    ok = ~np.isnan(P[:, 0])
    d = np.linalg.norm(z[:, None, :] - P[None, ok, :], axis=-1)
    pred = np.where(ok)[0][d.argmin(1)]
    return float((pred == y).mean())
