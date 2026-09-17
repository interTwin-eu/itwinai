from itwinai.torch.trainer import TorchTrainer

from .losses import relative_l2


class FNOTrainer(TorchTrainer):
    """FNO trainer. Identical to the stock trainer except for the relative L2 loss."""

    def create_model_loss_optimizer(self) -> None:
        super().create_model_loss_optimizer()
        # relative L2 is not among TrainingConfiguration's losses, so override it here,
        # after super() has built and distributed model, optimizer and scheduler.
        self.loss = relative_l2
