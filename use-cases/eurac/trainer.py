from typing import Dict, Any
import logging
from itwinai.components import Trainer, monitor_exec
from itwinai.loggers import WanDBLogger
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

from hython.models.lstm import CustomLSTM
from hython.train_val import train_val

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import nn

from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    DDPDistributedStrategy,
    HVDDistributedStrategy,
  #  DSDistributedStrategy,
)


def mse_metric(output, target, target_names = None): #target_names is added but unused for compatibility with a function in train_val
    metric_epoch = root_mean_squared_error(output, target)
    return metric_epoch

class LSTMTrainer(Trainer):
    def __init__(
        self,
        dynamic_names: list, 
        static_names : list, 
        target_names : list, 
        strategy_name: str, 
        shuffle_dataloader: bool,
        spatial_batch_size: int = None, 
        temporal_sampling_size: int = None, 
        seq_length : int = None, 
        hidden_size: int = None,  
        input_size : int = None,  #number of dynamic predic
        path2models: str = None, 
        epochs: int = None
    ):
        super().__init__()
        
        self.save_parameters(**self.locals2params(locals()))
        self.spatial_batch_size     = spatial_batch_size
        self.temporal_sampling_size = temporal_sampling_size
        self.target_names           = target_names
        self.static_names           = static_names
        self.path2models            = path2models
        self.epochs                 = epochs
        self.seq_length             = seq_length
        self.shuffle_dataloader     = bool(shuffle_dataloader)
      
        self.model_params={
            "input_size": input_size, #number of dynamic predictors - user_input
            "hidden_size": hidden_size, # user_input
            "output_size": len(target_names), # number_target - user_input
            "number_static_predictors": len(static_names), #number of static parameters - user_input 
            "target_names": target_names, 

        }

        # Instantiate Strategy
        self.is_distributed = True
        if strategy_name == 'ddp':
            if (not torch.cuda.is_available()
                    or not torch.cuda.device_count() > 1):
                raise RuntimeError('Resources unavailable')
            self.strategy = DDPDistributedStrategy(backend='nccl')
        elif strategy_name == 'horovod':
            self.strategy = HVDDistributedStrategy()
        elif strategy_name == 'deepspeed':
            self.strategy = DSDistributedStrategy(backend='nccl')
        elif strategy_name == 'None':
            self.strategy = None
            self.is_distributed = False
        else:
            raise NotImplementedError(
                f"Strategy {strategy} is not recognized/implemented.")


    @monitor_exec
    def execute(self, dataset, train_sampler, valid_sampler) -> None:

        #wandb
        wandb = WanDBLogger()

        if self.is_distributed:
            self.strategy.init()
            print(f"{foo}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} "
              f"{os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")
            device = strategy.dist_device()
        else:
            #setup device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logging.debug("Info:: Device set to", device)
        breakpoint()
        sampler=DistributedSampler(
            dataset,
            num_replicas=strategy.dist_gwsize(),
            rank=strategy.dist_grank(),
            shuffle=self.shuffle_dataloader
        )

        #setup data
        train_loader = DataLoader(train_data, batch_size=self.spatial_batch_size, sampler = sampler, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=self.spatial_batch_size, sampler = sampler, shuffle=False)
        logging.debug("Info: Data loaded to torch")  

        #model
        model = CustomLSTM(self.model_params)
        model = model.to(device)
        print(model)

        #optimizer
        opt = optim.Adam(model.parameters(), lr=1e-2)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=10)

        ## Set the training parameters
        params_train={
            "num_epochs": self.epochs,
            "temporal_sampling_size": self.temporal_sampling_size,
            "seq_length": self.seq_length,
            "ts_range": train_data.y.shape[1],
            "optimizer": opt,
            "loss_func": nn.MSELoss(),
            "metric_func": mse_metric,
            "train_dl": train_loader, 
            "val_dl"  : val_loader,
            "lr_scheduler": lr_scheduler,
            "path2weights": f"{self.path2models}/weights.pt", 
            "device": device,
            "target_names": self.target_names
        }
        logging.debug("Info: Model compiled")
        wandb.save_hyperparameters(params_train)

        #train
        model, sm_loss_history , sm_metric_history = train_val(model, params_train, wandb)      
        logging.debug("Info:: Model trained")

        # Extract the loss values
        for t in self.target_names: 
            train_loss = sm_metric_history[f'train_{t}']
            val_loss = sm_metric_history[f'val_{t}']

            # Create a list of epochs for the x-axis (e.g., [1, 2, 3, ..., 100])
            lepochs = list(range(1,params_train["num_epochs"] + 1))

            # Create the train and validation loss plots
            plt.figure(figsize=(10, 6))
            plt.plot(lepochs, train_loss, marker='o', linestyle='-', color='b', label='Training Loss')
            # plt.plot(lepochs, val_loss, marker='o', linestyle='-', color='r', label='Validation Loss')
            plt.title('Train Loss - SM')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"loss_{t}.png")

    def load_state(self):
        return super().load_state()

    def save_state(self):
        return super().save_state()