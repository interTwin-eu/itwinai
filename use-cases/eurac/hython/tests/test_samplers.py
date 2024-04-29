from functools import reduce
import numpy as np
import pytest

from hython.sampler import RegularIntervalSampler

model_domain = (100,100)
model_active = (10,90)

model_grid = np.empty(model_domain)
model_grid[:,:] = np.nan

xslice = slice(10,90)
yslice = slice(20,80)

model_grid[xslice, yslice] = np.arange(reduce(np.multiply,model_domain)).reshape(model_domain)[xslice, yslice]


def test_regular_shape():
    
    grid_55_00, _ = RegularIntervalSampler((5,5), (4,4)).sampling(model_grid)

    assert grid_55_00.shape == (20, 20)


def test_regular_idx():
    
    _, meta = RegularIntervalSampler((5,5), (0,0)).sampling(model_grid)
    
    index = meta.idx_sampled_1d
    
    assert index[0] == 100*5*(19 // 20)  + (0 % 20)*5
    assert index[9] == 100*5*(19 // 20)  + (9 % 20)*5
    assert index[19] == 100*5*(19 // 20) + (19 % 20)*5
    assert index[44] == 100*5*(44 // 20) + (44 % 20)*5
    assert index[69] == 100*5*(69 // 20) + (69 % 20)*5