"""Bearing fault simulator.

A train of impacts at the fault frequency, each exciting a single degree of
freedom resonance, on top of shaft harmonics and white noise.  Short and
readable rather than fast.
"""

import numpy as np

# Fault orders for the N205 / NU205 geometry used by the JNU dataset.
from features import ORDERS


def sdof_response(fs, fn=4000.0, decay_ms=1.0, dur_ms=6.0):
    """Impulse response of a single degree of freedom resonance."""
    t = np.arange(0, dur_ms / 1000.0, 1.0 / fs)
    beta = -np.log(0.01) / (decay_ms / 1000.0)
    return np.exp(-beta * t) * np.sin(2 * np.pi * fn * t)


def simulate(fault, fr, fs=25600, n=8192, snr_db=0.0, seed=0,
             q=0.0, slip=0.015, am=0.3, fn=4000.0, order=None):
    """One labelled vibration segment.

    fault : 'healthy', 'outer', 'inner' or 'ball'
    fr    : shaft rotation frequency in Hz
    q     : amplitude scaling exponent, 0 for pure dilation
    slip  : relative standard deviation of the impact jitter
    order : fault order in place of ORDERS[fault], or None for the default
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = np.zeros(n)

    # shaft harmonics, present in every class including healthy
    for m, amp in zip([1, 2, 3], [0.5, 0.25, 0.12]):
        x += amp * np.sin(2 * np.pi * m * fr * t + rng.uniform(0, 2 * np.pi))

    if fault != "healthy":
        fc = (ORDERS[fault] if order is None else order) * fr
        h = sdof_response(fs, fn=fn)
        a0 = 4.0 * (fr / 10.0) ** q
        k = 0
        while k / fc < n / fs:
            jitter = rng.normal(0.0, slip / fc)
            idx = int(round((k / fc + jitter) * fs))
            if 0 <= idx < n:
                amp = a0
                if fault in ("inner", "ball"):
                    amp *= 1 + am * np.cos(2 * np.pi * fr * idx / fs)
                end = min(idx + len(h), n)
                x[idx:end] += amp * h[: end - idx]
            k += 1

    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    x = x + rng.normal(0.0, np.sqrt(p_noise), n)
    return x
