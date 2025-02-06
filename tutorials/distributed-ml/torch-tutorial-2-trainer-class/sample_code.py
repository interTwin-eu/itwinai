# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""This file contains the sample code that was used for the snippets in the interTwin
presentation held on Feb. 18. These code snippets are meant as outlines for how to use
itwinai to simplify distributed ML.
"""

from itwinai.torch.distributed import TorchDDPStrategy
from itwinai.torch.trainer import TorchTrainer


# Included for the sake of linting
def train(model):
    pass


##############################################################################
# Using itwinai's Strategy but not the TorchTrainer
##############################################################################

# Create and initialize strategy
strategy = TorchDDPStrategy(backend="nccl")
strategy.init()

# Create dataset as usual
train_dataset = ...

# Use 'strategy' to create dataloader
train_dataloader = strategy.create_dataloader(train_dataset, ...)

# Create model, optimizer and scheduler as usual
model, optimizer, scheduler = ...

# Distribute them using 'strategy'
model, optimizer, scheduler = strategy.distributed(model, optimizer, scheduler)

# Train model as usual
train(model)  # Note: have to notify 'strategy' every time an epoch passes

# Clean up strategy at the end
strategy.clean_up()
##############################################################################


##############################################################################
# Using itwinai's TorchTrainer (which uses Strategy internally)
##############################################################################

# Create dataset as usual
train_dataset = ...

# Create model as usual
model = ...

trainer = TorchTrainer(config={}, model=model, strategy="ddp")

_, _, _, trained_model = trainer.execute(train_dataset, ...)
##############################################################################
