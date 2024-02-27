import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import xarray as xr
from sampler import AbstractSampler

from typing import List
from typing import Union, Any 


def preprocess(
            dynamic: xr.Dataset,
            static: xr.Dataset,
            target: xr.Dataset,
            dynamic_name: List,
            static_name: List, 
            target_name: List,
            sampler: AbstractSampler = None,
            return_sampler_meta: bool = False
              ) -> List[np.ndarray]:

    DIMS = {
        "orig": [ len(dynamic["lat"]), len(dynamic["lon"]), len(dynamic["time"])  ],
    }

    META = {"dyn":"", "static":"", "target":""}

    # select
    dyn_sel = dynamic[dynamic_name]
    static_sel = static[static_name]
    target_sel = target[target_name]

    # sampling

    if sampler:
        dyn_sel, dyn_sampler_meta = sampler.sampling(dyn_sel.transpose("lat", "lon", "time"))
        static_sel, static_sampler_meta = sampler.sampling(static_sel.transpose("lat", "lon"))
        target_sel, target_sampler_meta = sampler.sampling(target_sel.transpose("lat", "lon", "time"))

        DIMS["sampled_dims"] =  [ len(dyn_sel["lat"]), len(dyn_sel["lon"]), len(dyn_sel["time"])  ]

        print("sampling reduced dims (lat, lon): from ", DIMS["orig"][:2], " to ", DIMS["sampled_dims"][:2] )

        META.update({"dyn":dyn_sampler_meta,"static":static_sampler_meta, "target":target_sampler_meta})

    # train_test split 


    # reshape 
    Xd = ( dyn_sel
        .to_dataarray(dim="feat") # cast
        .stack(cell= ["lat","lon"]) # stack 
        .transpose("cell","time","feat") 
        )
    print("dynamic: ", Xd.shape, " => (GRIDCELL, TIME, FEATURE)")
    
    Xs = ( static_sel
    .drop_vars("spatial_ref")
    .to_dataarray(dim="feat")
    .stack(cell= ["lat","lon"])
    .transpose("cell","feat")
    )
    print("static: ", Xs.shape, " => (GRIDCELL, FEATURE)")

    Y = ( target_sel
        .to_dataarray(dim="feat")
        .stack(cell= ["lat","lon"])
        .transpose("cell","time", "feat")
        )
    print("target: ", Y.shape, " => (GRIDCELL, TIME, TARGET)")     



    return Xd.compute().values,Xs.compute().values, Y.compute().values, DIMS, META

def scale(a, how, axis, m1, m2):
    if how == 'standard':
        if m1 is None or m2 is None:
            m1, m2 = np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)
            
            m2[m2 == 0] = 1
            
            return (a - np.expand_dims(m1, axis = axis) )/ np.expand_dims(m2, axis = axis), m1, m2
        else:
            return (a - np.expand_dims(m2, axis = axis))/np.expand_dims(m2, axis = axis), None, None
    elif how == 'minmax':
        if m1 is None or m2 is None:
            m1, m2 = np.nanmin(a, axis=axis), np.nanmax(a, axis=axis)
            return (a - m1)/(m2 - m1), m1, m2
        else:
            return (a - m1)/(m2 - m1), None, None
                
def apply_normalization(a, type = "time", how='standard', m1=None, m2=None):
    """Assumes array of 
    dynamic: (gridcell, time, dimension)
    static: (gridcell, dimension)

    Parameters
    ----------
    a : 
        _description_
    type : str, optional
        , by default "space"
    how : str, optional
        _description_, by default 'standard'
    m1 : _type_, optional
        _description_, by default None
    m2 : _type_, optional
        _description_, by default None
    """
    if type == "time":
        return scale(a, how = how, axis = 1, m1 = m1, m2 = m2)
    elif type == "space": 
        return scale(a, how = how, axis = 0, m1 = m1,  m2 = m2)
    elif type == "spacetime":
         return scale(a, how = how, axis = (0, 1), m1 = m1, m2 = m2)
    else:
        raise NotImplementedError(f"Type {how} not implemented")



