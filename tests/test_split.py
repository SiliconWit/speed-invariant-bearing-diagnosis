"""The leakage test: training, validation and test data never overlap in time."""
import numpy as np
import pytest

import data


def test_split_is_by_time_not_by_window():
    """Blocks must be contiguous and disjoint in time."""
    x = np.arange(10000, dtype=float)
    parts = data.split_blocks(x)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    assert len(tr) + len(va) + len(te) == len(x), "the split must partition the recording"
    assert tr[-1] < va[0] < va[-1] < te[0], (
        "blocks must be contiguous in time and in order; if they are shuffled, "
        "overlapping windows land on both sides of the split")


@pytest.mark.needs_data("jnu")
def test_no_window_appears_in_two_splits():
    d = data.build("lfes")
    tr = {hash(r.tobytes()) for r in d["X_train"]}
    te = {hash(r.tobytes()) for r in d["X_test"]}
    assert not (tr & te), "identical windows appear in both train and test"
