"""Front ends: the log-frequency envelope spectrum and the linear-axis control.

The whole method rests on one identity.  A speed change by a factor rho dilates
the time axis, which scales every fault frequency by rho, which is a *translation*
by log(rho) on a logarithmic frequency axis.  A convolutional encoder is roughly
equivariant to translation, so putting the spectrum on a log axis converts a hard
problem (arbitrary rescaling) into an easy one (a shift).

The linear-axis squared envelope spectrum is kept as the control.  It carries the
same information; it just presents the speed change as a rescaling rather than a
shift.
"""
import numpy as np
from scipy.signal import hilbert

# N205 / NU205 geometry, the bearings in the JNU rig.
ORDERS = {"outer": 4.83, "inner": 7.17, "ball": 4.94}
CLASSES = ["healthy", "inner", "ball", "outer"]


def squared_envelope_spectrum(x, fs, pad=4):
    """Linear-axis squared envelope spectrum.

    `pad` zero-pads the FFT.  Without it the raw resolution is fs/N, which is
    coarser than the narrow low-frequency bands of the log axis, and the
    log-frequency vector comes out full of empty bands, which silently disables
    the front end.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    e2 = np.abs(hilbert(x)) ** 2
    e2 = e2 - e2.mean()
    nfft = int(len(x) * pad)
    spec = np.abs(np.fft.rfft(e2, n=nfft))
    freq = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freq, spec


def _bin_to_axis(freq, spec, edges):
    """Average spec into the given bands, interpolating bands too narrow to contain a bin."""
    centres = np.sqrt(edges[:-1] * edges[1:]) if np.all(edges > 0) else \
        0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(freq, edges) - 1
    out = np.interp(centres, freq, spec)
    counts = np.bincount(idx[(idx >= 0) & (idx < len(centres))],
                         minlength=len(centres))
    sums = np.bincount(idx[(idx >= 0) & (idx < len(centres))],
                       weights=spec[(idx >= 0) & (idx < len(centres))],
                       minlength=len(centres))
    wide = counts > 1
    out[wide] = sums[wide] / counts[wide]
    return out, centres


def lfes(x, fs, fmin=10.0, fmax=2000.0, nbins=384, gamma=1.0, pad=4):
    """Log-frequency envelope spectrum.  A speed change translates this vector."""
    freq, spec = squared_envelope_spectrum(x, fs, pad=pad)
    edges = np.geomspace(fmin, fmax, nbins + 1)
    out, centres = _bin_to_axis(freq, spec, edges)
    out = out / (out.mean() + 1e-12)          # keep harmonics visible
    out = np.log1p(gamma * out)               # compress the dynamic range
    n = np.linalg.norm(out)
    return (out / n if n > 0 else out), centres


def ses(x, fs, fmin=10.0, fmax=2000.0, nbins=384, gamma=1.0, pad=4):
    """Linear-frequency envelope spectrum, on the same number of bins.

    The control.  Identical processing except for the axis, so any difference
    between the two is attributable to the axis and not to the normalisation,
    the compression or the dimensionality.
    """
    freq, spec = squared_envelope_spectrum(x, fs, pad=pad)
    edges = np.linspace(fmin, fmax, nbins + 1)
    out, centres = _bin_to_axis(freq, spec, edges)
    out = out / (out.mean() + 1e-12)
    out = np.log1p(gamma * out)
    n = np.linalg.norm(out)
    return (out / n if n > 0 else out), centres


def order_spectrum(x, fs, fr, omax=12.0, nbins=384, gamma=1.0, pad=4):
    """Envelope spectrum resampled onto shaft orders.

    This is the strong baseline: if the shaft speed is known, dividing the
    frequency axis by it removes the dilation exactly.  It needs the shaft speed,
    and the log axis does not.
    """
    freq, spec = squared_envelope_spectrum(x, fs, pad=pad)
    edges = np.linspace(0.5, omax, nbins + 1) * fr
    out, _ = _bin_to_axis(freq, spec, edges)
    out = out / (out.mean() + 1e-12)
    out = np.log1p(gamma * out)
    n = np.linalg.norm(out)
    return (out / n if n > 0 else out), np.linspace(0.5, omax, nbins + 1)[:-1]


FRONT_ENDS = {"lfes": lfes, "ses": ses}
