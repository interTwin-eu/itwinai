from typing import Literal
import torch.nn as nn
import torch
import time
import numpy as np

from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
)

from src.model import Decoder, Decoder_2d_deep, UNet, GeneratorResNet
from src.utils import init_weights


from tqdm import tqdm


class NoiseGeneratorTrainer(TorchTrainer):

    def __init__(
        self,
        batch_size: int,
        learning_rate: float = 1e-3,
        num_epochs: int = 2,
        generator: Literal["simple", "deep", "resnet", "unet"] = "unet",
        loss: Literal["L1", "L2"] = "L1",
        name: str | None = None
    ) -> None:
        # Global configuration
        config = dict(
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            save_best=False,
            checkpoint_path="checkpoints/epoch_{}.pth"
        )
        super().__init__(config=config, epochs=num_epochs, name=name)
        self.save_parameters(**self.locals2params(locals()))
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._generator = generator
        self._loss = loss

    def create_model_loss_optimizer(self) -> None:
        # Select generator
        generator = self._generator.lower()
        if generator == "simple":
            self.model = Decoder(3, norm=False)
            init_weights(self.model, 'normal', scaling=.02)
        elif generator == "deep":
            self.model = Decoder_2d_deep(3)
            init_weights(self.model, 'normal', scaling=.02)
        elif generator == "resnet":
            self.model = GeneratorResNet(3, 12, 1)
            init_weights(self.model, 'normal', scaling=.01)
        elif generator == "unet":
            self.model = UNet(
                input_channels=3, output_channels=1, norm=False)
            init_weights(self.model, 'normal', scaling=.02)
        else:
            raise ValueError("Unrecognized generator type! Got", generator)

        # Select loss
        loss = self._loss.upper()
        if loss == "L1":
            self.loss = nn.L1Loss()
        elif loss == "L2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unrecognized loss type! Got", generator)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        else:
            distribute_kwargs = {}

        # Distributed model, optimizer, and scheduler
        self.model, self.optimizer, _ = self.strategy.distributed(
            self.model, self.optimizer, **distribute_kwargs
        )

    def train(self):
        # uncomment all lines relative to accuracy if you want to measure
        # IOU between generated and real spectrograms.
        # Note that it significantly slows down the whole process
        # it also might not work as the function has not been fully
        # implemented yet

        loss_plot = []
        val_loss_plot = []
        acc_plot = []
        val_acc_plot = []
        best_val_loss = 5000000
        for epoch in tqdm(range(1, self.num_epochs+1)):
            st = time.time()
            epoch_loss = []
            epoch_acc = []
            for i, batch in enumerate(self.train_dataloader):
                # batch= transform(batch)
                target = batch[:, 0].unsqueeze(1).to(self.device)
                # print(f'TARGET ON DEVICE: {target.get_device()}')
                target = target.float()
                input = batch[:, 1:].to(self.device)
                # print(f'INPUT ON DEVICE: {input.get_device()}')

                self.optimizer.zero_grad()
                generated = self.model(input.float())
                # generated=normalize_(generated,1)
                loss = self.loss(generated, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.detach().cpu().numpy())
                # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                # epoch_acc.append(acc)
            val_loss = []
            val_acc = []
            for batch in (self.validation_dataloader):
                # batch= transform(batch)
                target = batch[:, 0].unsqueeze(1).to(self.device)
                target = target.float()
                input = batch[:, 1:].to(self.device)
                with torch.no_grad():
                    generated = self.model(input.float())
                    # generated=normalize_(generated,1)
                    loss = self.loss(generated, target)
                    val_loss.append(loss.detach().cpu().numpy())
                    # acc=accuracy(generated.detach().cpu().numpy(),target.detach().cpu().numpy(),20)
                    # val_acc.append(acc)
            loss_plot.append(np.mean(epoch_loss))
            val_loss_plot.append(np.mean(val_loss))
            acc_plot.append(np.mean(epoch_acc))
            val_acc_plot.append(np.mean(val_acc))
            # print('epoch: {} loss: {} val loss: {} accuracy: {} val
            # accuracy: {}'.format(epoch,loss_plot[-1],val_loss_plot[-1],
            # acc_plot[-1],val_acc_plot[-1]))
            et = time.time()
            print('epoch: {} loss: {} val loss: {} time:{}s'.format(
                epoch, loss_plot[-1], val_loss_plot[-1], et-st))

            # Save checkpoint every 100 epochs
            if (epoch+1) % 100 == 0:
                # uncomment the following if you want to save checkpoint every
                # 100 epochs regardless of the performance of the model
                # checkpoint = {
                #     'epoch': epoch,
                #     'model_state_dict': generator.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss_plot[-1],
                #     'val_loss': val_loss_plot[-1],
                # }

                # checkpoint_filename = checkpoint_path.format(epoch)
                # torch.save(checkpoint, checkpoint_filename)

                # instead of val_loss and best_val loss we should
                # use accuracy!!!
                if self.config.save_best and val_loss_plot[-1] < best_val_loss:
                    # create checkpoint
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss_plot[-1],
                        'val_loss': val_loss_plot[-1],
                    }

                    # save checkpoint only if it is better than
                    # the previous ones
                    checkpoint_filename = self.config.checkpoint_path.format(
                        epoch)
                    torch.save(checkpoint, checkpoint_filename)

                    # update best model
                    best_val_loss = val_loss_plot[-1]
                    best_checkpoint_filename = (
                        self.config.checkpoint_path.format('best')
                    )
                    torch.save(checkpoint, best_checkpoint_filename)
        # return (loss_plot, val_loss_plot,
        # acc_plot, val_acc_plot ,acc_plot, val_acc_plot)
        return loss_plot, val_loss_plot, acc_plot, val_acc_plot
