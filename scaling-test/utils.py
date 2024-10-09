import pandas as pd
from typing import Tuple


def calculate_comp_and_comm_time(df: pd.DataFrame) -> Tuple[float, float]: 
  """Calculates the time spent computing and time spent communicating and 
  returns a tuple of these numbers in seconds

  Notes: 
    - Assumes that you are running with an NCCL backend. 
  """
  if not "name" in df.columns: 
    raise ValueError("DataFrame should contain column 'name', but does not!") 
  if not "self_cuda_time_total" in df.columns: 
    raise ValueError("DataFrame should contain column 'self_cuda_time_total', but does not!") 
  
  nccl_comm_pattern = r"ncclKernel_(?:AllReduce|Broadcast|Reduce|AllGather|ReduceScatter)"
  aten_comp_pattern = r"aten::"

  comm_df = df[df['name'].str.contains(nccl_comm_pattern)]
  comp_df = df[df['name'].str.contains(aten_comp_pattern)]

  comp_time = comp_df["self_cuda_time_total"].sum()
  comm_time = comm_df["self_cuda_time_total"].sum()
  
  # Converting from microseconds to seconds
  comp_time *= 1e-6
  comm_time *= 1e-6
  
  return comp_time, comm_time

