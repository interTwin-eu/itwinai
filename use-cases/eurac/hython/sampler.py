import torch 
import numpy as np
import xarray as xr

from abc import ABC, abstractmethod
from dataclasses import dataclass

# type hints
from typing import Any, Tuple, List
from numpy.typing import NDArray


@dataclass
class SamplerMetaData:
    """Store metadata to restructure original grid from the sampled grid
    """
    idx_orig_2d: NDArray
    idx_sampled_1d: NDArray 


class AbstractSampler(ABC):
    
    def __init__(self):
        """Pass parametes required by the sampling approach
        """    
        pass

    # def __post_init__(self):
    #     self._has_required_attributes()

    # def _has_required_attributes(self):
    #     req_attrs: List[str] = ['grid']
    #     for attr in req_attrs:
    #         if not hasattr(self, attr):
    #             raise AttributeError(f"Missing attribute: '{attr}'")
        
    @abstractmethod
    def sampling(self, grid: NDArray | xr.DataArray | xr.Dataset) -> Tuple[NDArray, SamplerMetaData]:
        """Sample the original grid. Must be instantiated by a concrete class that implements the sampling approach.

        Args:
            grid (NDArray | xr.DataArray | xr.Dataset): The gridded data to be sampled

        Returns:
            Tuple[NDArray, SamplerMetaData]: The sampled grid and sampler's metadata
        """
        
        pass

class RegularIntervalSampler(AbstractSampler):

    def __init__(self,
                intervals: tuple[int] = (5,5), 
                origin: tuple[int] = (0, 0)):
        
        self.intervals = intervals
        self.origin = origin

        if intervals[0] != intervals[1]:
            raise NotImplementedError("Different x,y intervals not yet implemented!")

        if origin[0] != origin[1]:
            raise NotImplementedError("Different x,y origins not yet implemented!")

    def sampling(self, grid):
        
        """Sample a N-dimensional array by regularly-spaced points along the spatial axes. 

        Parameters
        ----------
        grid : np.ndarray
            Spatial axes should be the first 2 dimensions, i.e. (lat, lon) or (y, x)
        intervals : tuple[int], optional
            Sampling intervals in CRS distance, by default (5,5).
            5,5 in a 1 km resolution grid, means sampling every 5 km in x and y directions.
        origin : tuple[int], optional
            _description_, by default (0, 0)

        Returns
        -------
        np.ndarray
            _description_
        """

        if isinstance(grid, np.ndarray):
            shape = grid.shape
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            shape = (len(grid.lat), len(grid.lon))
        else:
            pass
        
        ishape,iorigin,iintervals = shape[0], self.origin[0], self.intervals[0] # rows (y, lat)
        jshape,jorigin,jintervals = shape[1], self.origin[1], self.intervals[1] # columns (x, lon)

        irange = np.arange(iorigin, ishape, iintervals)
        jrange = np.arange(jorigin, jshape, jintervals)

        grid_idx = np.arange(0, ishape * jshape, 1).reshape(ishape, jshape)

        idx_sampled = grid_idx[irange[:,None], jrange].flatten()

        if isinstance(grid, np.ndarray):
            samples = grid[irange[:, None], jrange]
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            samples = grid.isel(lat=irange, lon=jrange)
        else:
            pass
        
        
        return samples, SamplerMetaData(idx_orig_2d = grid_idx, idx_sampled_1d = idx_sampled)



class StratifiedSampler(AbstractSampler):
    pass



class SpatialCorrSampler(AbstractSampler):
    pass

