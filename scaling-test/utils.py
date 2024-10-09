from torchvision import datasets, transforms
import pandas as pd
from typing import Tuple

def imagenet_dataset(data_root: str):
    """Create a torch dataset object for Imagenet."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imagenet = datasets.ImageFolder(
        root=data_root,
        transform=transform
    )
    return imagenet 

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

