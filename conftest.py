"""Put code/ on the import path, and skip tests whose recordings are absent.

A test marked `@pytest.mark.needs_data("jnu")` or `needs_data("cwru")` reads the
recordings.  When they are not present, as on CI, it is skipped with the reason.
Fetch them with `python3 data/fetch_data.py`.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))


def _show(folder):
    rel = os.path.relpath(folder, ROOT)
    return folder if rel.startswith("..") else rel


def _missing(name):
    import paths
    if name == "jnu":
        import data
        folder, files = paths.JNU, sorted(data.FILES.values())
    elif name == "cwru":
        import cwru
        folder, files = paths.CWRU, [f"{n}.mat" for n in sorted(cwru.FILES)]
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return folder, [f for f in files if not os.path.exists(os.path.join(folder, f))]


def pytest_runtest_setup(item):
    for mark in item.iter_markers("needs_data"):
        for name in mark.args:
            folder, missing = _missing(name)
            if missing:
                pytest.skip(f"{name.upper()} recordings not found in {_show(folder)} "
                            f"({len(missing)} missing); run python3 data/fetch_data.py")
