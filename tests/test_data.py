"""The recordings: provenance and basic facts about them."""
import os
import re
import sys

import numpy as np
import pytest

import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
import fetch_data  # noqa: E402


def test_manifest_matches_fetch_script():
    """data/MANIFEST.md and data/fetch_data.py must list the same files and checksums."""
    text = open(os.path.join(ROOT, "data", "MANIFEST.md")).read()
    rows = re.findall(r"^\| (cwru|jnu)/(\S+) \| \d+ \| ([0-9a-f]{32}) \|$", text, re.M)
    listed = {(sub, name): h for sub, name, h in rows}
    assert listed == fetch_data.MD5


def test_code_uses_exactly_the_fetched_files():
    import data, cwru
    used = {("jnu", f) for f in data.FILES.values()}
    used |= {("cwru", f"{n}.mat") for n in cwru.FILES}
    assert used == set(fetch_data.MD5)


@pytest.mark.needs_data("jnu", "cwru")
def test_checksums():
    for (sub, name), want in fetch_data.MD5.items():
        path = os.path.join(paths.DATA, sub, name)
        assert fetch_data.md5(path) == want, f"{sub}/{name} differs from the manifest"


@pytest.mark.needs_data("jnu")
def test_jnu_recording_lengths():
    """Fault recordings are 10 s and normal recordings 30 s at 50 kHz."""
    import data
    for (cls, rpm), fname in data.FILES.items():
        n = len(np.loadtxt(os.path.join(paths.JNU, fname)))
        want = 30 if cls == "healthy" else 10
        assert abs(n / data.FS - want) < 0.5, f"{fname}: {n / data.FS:.2f} s"


@pytest.mark.needs_data("cwru")
def test_cwru_speed_range_is_small():
    """The four CWRU conditions span under 5 % in recorded shaft speed."""
    from scipy.io import loadmat
    import cwru
    rpm = []
    for num in cwru.FILES:
        d = loadmat(os.path.join(paths.CWRU, f"{num}.mat"))
        rpm += [float(d[k].ravel()[0]) for k in d if k.endswith("RPM")]
    assert rpm, "no RPM field found in any file"
    assert max(rpm) / min(rpm) < 1.05
