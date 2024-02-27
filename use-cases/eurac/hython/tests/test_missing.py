from functools import reduce
import numpy as np
import pytest

from hython.sampler import missing_gridcell_index, reconstruct

# (gridcell, time, dimension)
a_orig = np.random.randint(0, 100, (1000, 50, 3)).astype(np.float64)

static = np.random.randint(0,100, (1000, 3)).astype(np.float64)

static[static == 10] = np.nan

def test_missing():
    
    idx = missing_gridcell_index(static)

    a_clean = a_orig[~idx]

    a_rec = reconstruct(a_clean, a_orig.shape, idx)

    assert a_rec.shape == a_orig.shape
    assert np.allclose(a_rec, a_orig)
