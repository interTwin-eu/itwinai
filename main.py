# Without TorchTrainer but With Strategy
from itwinai.torch.distributed import TorchDDPStrategy

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
train(model) # Note: have to notify 'strategy' every time an epoch passes

# Clean up strategy at the end
strategy.clean_up()


# With TorchTrainer
from itwinai.torch.trainer import TorchTrainer

# Create dataset as usual
train_dataset = ...

# Create model as usual
model = ...

trainer = TorchTrainer(config={}, model=model, strategy="ddp")

_, _, _, trained_model = trainer.execute(train_dataset, ...)































