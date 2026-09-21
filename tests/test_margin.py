"""The margin geometry and the encoder, on synthetic embeddings."""
import numpy as np

import margin


def _clusters(seed=0, n=50, k=8, spread=0.05):
    rng = np.random.default_rng(seed)
    centres = np.eye(4, k)
    z = np.concatenate([c + spread * rng.standard_normal((n, k)) for c in centres])
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z, np.repeat(np.arange(4), n)


def test_prototypes_are_unit_length_and_separated():
    z, y = _clusters()
    P = margin.prototypes(z, y)
    assert np.allclose(np.linalg.norm(P, axis=1), 1.0)
    assert abs(margin.separation(P) - np.sqrt(2)) < 0.05, "orthogonal classes sit sqrt(2) apart"


def test_margin_ratio_definition():
    assert margin.margin_ratio(1.0, 0.25, 0.25) == 1.0
    assert margin.margin_ratio(2.0, 0.25, 0.25) == 2.0


def test_condition_holds_implies_correct_nearest_prototype():
    """Proposition 1: with r + delta < Delta/2 every displaced sample stays correct."""
    z, y = _clusters(spread=0.02)
    g = margin.geometry(z, y, z, y)
    P = g["proto"]
    rng = np.random.default_rng(1)
    step = rng.standard_normal(z.shape)
    step *= 0.2 / np.linalg.norm(step, axis=1, keepdims=True)
    d = 0.2
    assert g["Delta"] / (2 * (g["r_bar"] + d)) > 1.0
    assert margin.nearest_prototype_acc(z + step, y, P) == 1.0


def test_paired_displacement_is_zero_for_identical_embeddings():
    z, _ = _clusters()
    assert margin.displacement_paired(z, z) == 0.0


def test_encoder_output_is_on_the_unit_sphere():
    import torch
    from models import Net
    torch.manual_seed(0)
    logits, z = Net(nbins=256, k=64, n_class=4)(torch.randn(5, 256))
    assert logits.shape == (5, 4) and z.shape == (5, 64)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)
