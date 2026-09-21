"""Properties the front ends must satisfy, independent of any dataset."""
import numpy as np
import pytest

import features

FS, N = 50000.0, 8192


def _noise(seed=0):
    return np.random.default_rng(seed).standard_normal(N)


def test_unit_norm_and_dimension():
    for fn in (features.lfes, features.ses):
        v, c = fn(_noise(), FS, nbins=256)
        assert v.shape == (256,), "front end must return the requested number of bins"
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6, "vectors must be unit length"
        assert len(c) == 256


def test_no_empty_bands():
    """The failure that silently disables the whole method."""
    v, _ = features.lfes(_noise(), FS, fmin=20.0, fmax=2000.0, nbins=256, pad=8)
    assert np.count_nonzero(v) == 256, (
        "some log bands are empty: they are narrower than the FFT resolution. "
        "Zero-pad the FFT and interpolate bands that contain no bin.")


def test_axes_differ_only_in_spacing():
    """Both front ends must agree on everything except where the bands sit."""
    x = _noise()
    a, ca = features.lfes(x, FS, nbins=256)
    b, cb = features.ses(x, FS, nbins=256)
    assert a.shape == b.shape
    assert not np.allclose(ca, cb), "the two axes must actually differ"
    assert abs(np.linalg.norm(a) - np.linalg.norm(b)) < 1e-6


def test_dilation_is_a_translation_on_the_log_axis():
    """The identity the method rests on.

    Generate one signal, read it at two rates, and check that the log-axis
    spectrum has moved along the axis rather than changed shape. The size of the
    shift against the predicted log(rho)/bin width is measured by run_controls.py;
    here we only require that a shift exists and that it is in the right direction.
    """
    import sim
    rho = 1.5
    x = sim.simulate("outer", 10.0, fs=25600, n=int(N * 2), snr_db=10, seed=0)
    a = x[:N]
    b = np.interp(np.arange(N) * rho, np.arange(len(x)), x)
    va, _ = features.lfes(a, 25600.0, nbins=256)
    vb, _ = features.lfes(b, 25600.0, nbins=256)
    best = min(range(-90, 91), key=lambda k: np.linalg.norm(np.roll(va, k) - vb))
    assert best > 0, "a speed increase must shift the spectrum towards higher frequency"
    assert np.linalg.norm(np.roll(va, best) - vb) < np.linalg.norm(va - vb), (
        "shifting must bring the two closer together; if it does not, the two "
        "signals are probably independent realisations rather than one signal "
        "read at two rates")
