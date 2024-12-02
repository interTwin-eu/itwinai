# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Kalliopi Tsolaki
#
# Credit:
# - Kalliopi Tsolaki <kalliopi.tsolaki@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import math
import os
import sys
from collections import defaultdict
from typing import Any

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itwinai.loggers import Logger as BaseItwinaiLogger


class Generator(nn.Module):
    def __init__(self, latent_dim):  # img_shape
        super().__init__()
        # self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.l1 = nn.Linear(self.latent_dim, 5184)
        self.up1 = nn.Upsample(scale_factor=(6, 6, 6), mode="trilinear", align_corners=False)
        self.conv1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(6, 6, 8), padding=0)
        nn.init.kaiming_uniform_(self.conv1.weight)
        # num_features is the number of channels (see doc)
        self.bn1 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.pad1 = nn.ConstantPad3d((1, 1, 2, 2, 2, 2), 0)

        self.conv2 = nn.Conv3d(in_channels=8, out_channels=6, kernel_size=(4, 4, 6), padding=0)
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm3d(num_features=6, eps=1e-6)
        self.pad2 = nn.ConstantPad3d((1, 1, 2, 2, 2, 2), 0)

        self.conv3 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(4, 4, 6), padding=0)
        nn.init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm3d(num_features=6, eps=1e-6)
        self.pad3 = nn.ConstantPad3d((1, 1, 2, 2, 2, 2), 0)

        self.conv4 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(4, 4, 6), padding=0)
        nn.init.kaiming_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm3d(num_features=6, eps=1e-6)
        self.pad4 = nn.ConstantPad3d((0, 0, 1, 1, 1, 1), 0)

        self.conv5 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(3, 3, 5), padding=0)
        nn.init.kaiming_uniform_(self.conv5.weight)
        self.bn5 = nn.BatchNorm3d(num_features=6, eps=1e-6)
        self.pad5 = nn.ConstantPad3d((0, 0, 1, 1, 1, 1), 0)

        self.conv6 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(3, 3, 3), padding=0)
        nn.init.kaiming_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv3d(in_channels=6, out_channels=1, kernel_size=(2, 2, 2), padding=0)
        nn.init.xavier_normal_(self.conv7.weight)

    def forward(self, z):
        img = self.l1(z)
        img = img.view(-1, 8, 9, 9, 8)
        img = self.up1(img)
        img = self.conv1(img)
        img = F.relu(img)
        img = self.bn1(img)
        img = self.pad1(img)

        img = self.conv2(img)
        img = F.relu(img)
        img = self.bn2(img)
        img = self.pad2(img)

        img = self.conv3(img)
        img = F.relu(img)
        img = self.bn3(img)
        img = self.pad3(img)

        img = self.conv4(img)
        img = F.relu(img)
        img = self.bn4(img)
        img = self.pad4(img)

        img = self.conv5(img)
        img = F.relu(img)
        img = self.bn5(img)
        img = self.pad5(img)

        img = self.conv6(img)
        img = F.relu(img)

        img = self.conv7(img)
        img = F.relu(img)

        return img


class Discriminator(nn.Module):
    def __init__(self, power):
        super().__init__()

        self.power = power

        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=16, kernel_size=(5, 6, 6), padding=(2, 3, 3)
        )
        self.drop1 = nn.Dropout(0.2)
        self.pad1 = nn.ConstantPad3d((1, 1, 0, 0, 0, 0), 0)

        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=8, kernel_size=(5, 6, 6), padding=0
        )
        self.bn1 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.drop2 = nn.Dropout(0.2)
        self.pad2 = nn.ConstantPad3d((1, 1, 0, 0, 0, 0), 0)

        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(5, 6, 6), padding=0)
        self.bn2 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.drop3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(5, 6, 6), padding=0)
        self.bn3 = nn.BatchNorm3d(num_features=8, eps=1e-6)
        self.drop4 = nn.Dropout(0.2)

        self.avgpool = nn.AvgPool3d((2, 2, 2))
        self.flatten = nn.Flatten()

        # The input features for the Linear layer need to be calculated based
        # on the output shape from the previous layers.
        self.fakeout = nn.Linear(19152, 1)
        self.auxout = nn.Linear(19152, 1)  # The same as above for this layer.

    # calculate sum of intensities
    def ecal_sum(self, image, daxis):
        sum = torch.sum(image, dim=daxis)
        return sum

    # angle calculation
    def ecal_angle(self, image, daxis1):
        image = torch.squeeze(image, dim=daxis1)  # squeeze along channel axis

        # get shapes
        x_shape = image.shape[1]
        y_shape = image.shape[2]
        z_shape = image.shape[3]
        sumtot = torch.sum(image, dim=(1, 2, 3))  # sum of events

        # get 1. where event sum is 0 and 0 elsewhere
        amask = torch.where(sumtot == 0.0, torch.ones_like(sumtot), torch.zeros_like(sumtot))
        # masked_events = torch.sum(amask)  # counting zero sum events

        # ref denotes barycenter as that is our reference point
        x_ref = torch.sum(
            torch.sum(image, dim=(2, 3))
            * (
                torch.arange(x_shape, device=image.device, dtype=torch.float32).unsqueeze(0)
                + 0.5
            ),
            dim=1,
        )  # sum for x position * x index
        y_ref = torch.sum(
            torch.sum(image, dim=(1, 3))
            * (
                torch.arange(y_shape, device=image.device, dtype=torch.float32).unsqueeze(0)
                + 0.5
            ),
            dim=1,
        )
        z_ref = torch.sum(
            torch.sum(image, dim=(1, 2))
            * (
                torch.arange(z_shape, device=image.device, dtype=torch.float32).unsqueeze(0)
                + 0.5
            ),
            dim=1,
        )

        # return max position if sumtot=0 and divide by sumtot otherwise
        x_ref = torch.where(sumtot == 0.0, torch.ones_like(x_ref), x_ref / sumtot)
        y_ref = torch.where(sumtot == 0.0, torch.ones_like(y_ref), y_ref / sumtot)
        z_ref = torch.where(sumtot == 0.0, torch.ones_like(z_ref), z_ref / sumtot)

        # reshape
        x_ref = x_ref.unsqueeze(1)
        y_ref = y_ref.unsqueeze(1)
        z_ref = z_ref.unsqueeze(1)

        sumz = torch.sum(image, dim=(1, 2))  # sum for x,y planes going along z

        # Get 0 where sum along z is 0 and 1 elsewhere
        zmask = torch.where(sumz == 0.0, torch.zeros_like(sumz), torch.ones_like(sumz))

        x = torch.arange(x_shape, device=image.device).unsqueeze(0)  # x indexes
        x = (x.unsqueeze(2).float()) + 0.5
        y = torch.arange(y_shape, device=image.device).unsqueeze(0)  # y indexes
        y = (y.unsqueeze(2).float()) + 0.5

        # barycenter for each z position
        x_mid = torch.sum(torch.sum(image, dim=2) * x, dim=1)
        y_mid = torch.sum(torch.sum(image, dim=1) * y, dim=1)

        x_mid = torch.where(
            sumz == 0.0, torch.zeros_like(sumz), x_mid / sumz
        )  # if sum != 0 then divide by sum
        y_mid = torch.where(
            sumz == 0.0, torch.zeros_like(sumz), y_mid / sumz
        )  # if sum != 0 then divide by sum

        # Angle Calculations
        z = (
            torch.arange(
                z_shape,
                device=image.device,
                dtype=torch.float32,
                # Make an array of z indexes for all events
            )
            + 0.5
        ) * torch.ones_like(z_ref)

        # projection from z axis with stability check
        zproj = torch.sqrt(
            torch.max(
                (x_mid - x_ref) ** 2.0 + (z - z_ref) ** 2.0,
                torch.tensor([torch.finfo(torch.float32).eps]).to(x_mid.device),
            )
        )
        # torch.finfo(torch.float32).eps))
        # to avoid divide by zero for zproj =0
        m = torch.where(zproj == 0.0, torch.zeros_like(zproj), (y_mid - y_ref) / zproj)
        m = torch.where(z < z_ref, -1 * m, m)  # sign inversion
        ang = (math.pi / 2.0) - torch.atan(m)  # angle correction
        zmask = torch.where(zproj == 0.0, torch.zeros_like(zproj), zmask)
        ang = ang * zmask  # place zero where zsum is zero
        ang = ang * z  # weighted by position
        sumz_tot = z * zmask  # removing indexes with 0 energies or angles

        # zunmasked = K.sum(zmask, axis=1) # used for simple mean
        # Mean does not include positions where zsum=0
        # ang = K.sum(ang, axis=1)/zunmasked

        # sum ( measured * weights)/sum(weights)
        ang = torch.sum(ang, dim=1) / torch.sum(sumz_tot, dim=1)
        # Place 100 for measured angle where no energy is deposited in events
        ang = torch.where(amask == 0.0, ang, 100.0 * torch.ones_like(ang))
        ang = ang.unsqueeze(1)
        return ang

    def forward(self, x):
        z = self.conv1(x)
        z = F.leaky_relu(z)
        z = self.drop1(z)
        z = self.pad1(z)

        z = self.conv2(z)
        z = F.leaky_relu(z)
        z = self.bn1(z)
        z = self.drop2(z)
        z = self.pad2(z)

        z = self.conv3(z)
        z = F.leaky_relu(z)
        z = self.bn2(z)
        z = self.drop3(z)

        z = self.conv4(z)
        z = F.leaky_relu(z)
        z = self.bn3(z)
        z = self.drop4(z)
        z = self.avgpool(z)
        z = self.flatten(z)

        # generation output that says fake/real
        fake = torch.sigmoid(self.fakeout(z))
        aux = self.auxout(z)  # auxiliary output
        inv_image = x.pow(1.0 / self.power)
        ang = self.ecal_angle(inv_image, 1)  # angle calculation
        ecal = self.ecal_sum(inv_image, (2, 3, 4))  # sum of energies

        return fake, aux, ang, ecal


class ThreeDGAN(pl.LightningModule):
    def __init__(
        self,
        latent_size=256,
        loss_weights=[3, 0.1, 25, 0.1],
        power=0.85,
        lr=0.001,
        checkpoints_dir: str = ".",
        provenance_verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.latent_size = latent_size
        self.loss_weights = loss_weights
        self.lr = lr
        self.power = power
        self.checkpoints_dir = checkpoints_dir
        self.provenance_verbose = provenance_verbose
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.generator = Generator(self.latent_size)
        self.discriminator = Discriminator(self.power)

        self.epoch_gen_loss = []
        self.epoch_disc_loss = []
        self.disc_epoch_test_loss = []
        self.gen_epoch_test_loss = []
        self.index = 0
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        # self.pklfile = checkpoint_path
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        # os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def itwinai_logger(self) -> BaseItwinaiLogger:
        try:
            itwinai_logger = self.trainer.itwinai_logger
        except AttributeError:
            print("WARNING: itwinai_logger attribute not set " f"in {self.__class__.__name__}")
            itwinai_logger = None
        return itwinai_logger

    def on_fit_start(self) -> None:
        if self.itwinai_logger:
            # Log hyper-parameters
            self.itwinai_logger.save_hyperparameters(self.hparams)

    def BitFlip(self, x, prob=0.05):
        """
        Flips a single bit according to a certain probability.

        Args:
            x (list): list of bits to be flipped
            prob (float): probability of flipping one bit

        Returns:
            list: List of flipped bits

        """
        x = np.array(x)
        selection = np.random.uniform(0, 1, x.shape) < prob
        x[selection] = 1 * np.logical_not(x[selection])
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-7))) * 100

    def compute_global_loss(self, labels, predictions, loss_weights=(3, 0.1, 25, 0.1)):
        # Can be initialized outside
        binary_crossentropy_object = nn.BCEWithLogitsLoss(reduction="none")
        # there is no equivalent in pytorch for
        # tf.keras.losses.MeanAbsolutePercentageError --> using the
        # custom "mean_absolute_percentage_error" above!
        mean_absolute_percentage_error_object1 = self.mean_absolute_percentage_error(
            predictions[1], labels[1]
        )
        mean_absolute_percentage_error_object2 = self.mean_absolute_percentage_error(
            predictions[3], labels[3]
        )
        mae_object = nn.L1Loss(reduction="none")

        binary_example_loss = (
            binary_crossentropy_object(predictions[0], labels[0]) * loss_weights[0]
        )

        # mean_example_loss_1 = mean_absolute_percentage_error_object(
        # predictions[1], labels[1]) * loss_weights[1]
        mean_example_loss_1 = mean_absolute_percentage_error_object1 * loss_weights[1]

        mae_example_loss = mae_object(predictions[2], labels[2]) * loss_weights[2]

        # mean_example_loss_2 = mean_absolute_percentage_error_object(
        # predictions[3], labels[3]) * loss_weights[3]
        mean_example_loss_2 = mean_absolute_percentage_error_object2 * loss_weights[3]

        binary_loss = binary_example_loss.mean()
        mean_loss_1 = mean_example_loss_1.mean()
        mae_loss = mae_example_loss.mean()
        mean_loss_2 = mean_example_loss_2.mean()

        return [binary_loss, mean_loss_1, mae_loss, mean_loss_2]

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        image_batch, energy_batch, ang_batch, ecal_batch = (
            batch["X"],
            batch["Y"],
            batch["ang"],
            batch["ecal"],
        )

        image_batch = image_batch.permute(0, 4, 1, 2, 3)

        image_batch = image_batch.to(self.device)
        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)
        ecal_batch = ecal_batch.to(self.device)

        optimizer_discriminator, optimizer_generator = self.optimizers()
        batch_size = energy_batch.shape[0]

        noise = torch.randn(
            (batch_size, self.latent_size - 2), dtype=torch.float32, device=self.device
        )
        # print(f'Energy elements: {energy_batch.numel} {energy_batch.shape}')
        # print(f'Angle elements: {ang_batch.numel} {ang_batch.shape}')
        generator_ip = torch.cat(
            (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise), dim=1
        )
        generated_images = self.generator(generator_ip)

        # Train discriminator first on real batch
        fake_batch = self.BitFlip(np.ones(batch_size).astype(np.float32))
        fake_batch = torch.tensor([[el] for el in fake_batch]).to(self.device)
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        predictions = self.discriminator(image_batch)
        # print("calculating real_batch_loss...")
        real_batch_loss = self.compute_global_loss(labels, predictions, self.loss_weights)
        if self.itwinai_logger:
            self.itwinai_logger.log(
                item=sum(real_batch_loss),
                identifier="real_batch_loss",
                kind="metric",
                step=self.global_step,
                batch_idx=batch_idx,
                context="training",
            )

        # self.log("real_batch_loss", sum(real_batch_loss),
        #          prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # print("real batch disc train")
        # the following 3 lines correspond in tf version to:
        # gradients = tape.gradient(real_batch_loss,
        # discriminator.trainable_variables)
        # optimizer_discriminator.apply_gradients(zip(gradients,
        #  discriminator.trainable_variables)) in Tensorflow
        optimizer_discriminator.zero_grad()
        self.manual_backward(sum(real_batch_loss))
        # sum(real_batch_loss).backward()
        # real_batch_loss.backward()
        optimizer_discriminator.step()

        # Train discriminator on the fake batch
        fake_batch = self.BitFlip(np.zeros(batch_size).astype(np.float32))
        fake_batch = torch.tensor([[el] for el in fake_batch]).to(self.device)
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        predictions = self.discriminator(generated_images)

        fake_batch_loss = self.compute_global_loss(labels, predictions, self.loss_weights)
        # self.log("fake_batch_loss", sum(fake_batch_loss),
        #          prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.itwinai_logger:
            self.itwinai_logger.log(
                item=sum(fake_batch_loss),
                identifier="fake_batch_loss",
                kind="metric",
                step=self.global_step,
                batch_idx=batch_idx,
                context="training",
            )

        # print("fake batch disc train")
        # the following 3 lines correspond to
        # gradients = tape.gradient(fake_batch_loss,
        # discriminator.trainable_variables)
        # optimizer_discriminator.apply_gradients(zip(gradients,
        # discriminator.trainable_variables)) in Tensorflow
        optimizer_discriminator.zero_grad()
        self.manual_backward(sum(fake_batch_loss))
        # sum(fake_batch_loss).backward()
        optimizer_discriminator.step()

        # avg_disc_loss = (sum(real_batch_loss) + sum(fake_batch_loss)) / 2

        trick = np.ones(batch_size).astype(np.float32)
        fake_batch = torch.tensor([[el] for el in trick]).to(self.device)
        labels = [fake_batch, energy_batch.view(-1, 1), ang_batch, ecal_batch]

        gen_losses_train = []
        # Train generator twice using combined model
        for _ in range(2):
            noise = torch.randn((batch_size, self.latent_size - 2)).to(self.device)
            generator_ip = torch.cat(
                (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise), dim=1
            )

            generated_images = self.generator(generator_ip)
            predictions = self.discriminator(generated_images)

            loss = self.compute_global_loss(labels, predictions, self.loss_weights)
            # self.log("gen_loss", sum(loss), prog_bar=True,
            #          on_step=True, on_epoch=True, sync_dist=True)

            if self.itwinai_logger:
                self.itwinai_logger.log(
                    item=sum(loss),
                    identifier="gen_loss",
                    kind="metric",
                    step=self.global_step,
                    batch_idx=batch_idx,
                    context="training",
                )

            # print("gen train")
            optimizer_generator.zero_grad()
            self.manual_backward(sum(loss))
            # sum(loss).backward()
            optimizer_generator.step()

            for el in loss:
                gen_losses_train.append(el)

        avg_generator_loss = sum(gen_losses_train) / len(gen_losses_train)
        # self.log("generator_loss", avg_generator_loss.item(),
        #          prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.itwinai_logger:
            self.itwinai_logger.log(
                item=avg_generator_loss.item(),
                identifier="generator_loss",
                kind="metric",
                step=self.global_step,
                batch_idx=batch_idx,
                context="training",
            )
            # Log provenance information
            if self.provenance_verbose:
                # Log provenance at every training step
                self._log_provenance(context="training")

        # avg_generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses_train)]
        # self.log("generator_loss", sum(avg_generator_loss), prog_bar=True,
        # on_step=True, on_epoch=True, sync_dist=True)

        gen_losses = []
        # I'm not returning anything as in pl you do not return anything when
        # you back-propagate manually
        # return_loss = real_batch_loss
        real_batch_loss = [
            real_batch_loss[0],
            real_batch_loss[1],
            real_batch_loss[2],
            real_batch_loss[3],
        ]
        fake_batch_loss = [
            fake_batch_loss[0],
            fake_batch_loss[1],
            fake_batch_loss[2],
            fake_batch_loss[3],
        ]
        gen_batch_loss = [
            gen_losses_train[0],
            gen_losses_train[1],
            gen_losses_train[2],
            gen_losses_train[3],
        ]
        gen_losses.append(gen_batch_loss)
        gen_batch_loss = [
            gen_losses_train[4],
            gen_losses_train[5],
            gen_losses_train[6],
            gen_losses_train[7],
        ]
        gen_losses.append(gen_batch_loss)

        real_batch_loss = [el.cpu().detach().numpy() for el in real_batch_loss]
        real_batch_loss_total_loss = np.sum(real_batch_loss)
        new_real_batch_loss = [real_batch_loss_total_loss]
        for i_weights in range(len(real_batch_loss)):
            new_real_batch_loss.append(
                real_batch_loss[i_weights] / self.loss_weights[i_weights]
            )
        real_batch_loss = new_real_batch_loss

        fake_batch_loss = [el.cpu().detach().numpy() for el in fake_batch_loss]
        fake_batch_loss_total_loss = np.sum(fake_batch_loss)
        new_fake_batch_loss = [fake_batch_loss_total_loss]
        for i_weights in range(len(fake_batch_loss)):
            new_fake_batch_loss.append(
                fake_batch_loss[i_weights] / self.loss_weights[i_weights]
            )
        fake_batch_loss = new_fake_batch_loss

        # if ecal sum has 100% loss(generating empty events) then end
        # the training
        if fake_batch_loss[3] == 100.0 and self.index > 10:
            # print("Empty image with Ecal loss equal to 100.0 "
            #       f"for {self.index} batch")
            torch.save(
                self.generator.state_dict(),
                os.path.join(self.checkpoints_dir, "generator_weights.pth"),
            )
            torch.save(
                self.discriminator.state_dict(),
                os.path.join(self.checkpoints_dir, "discriminator_weights.pth"),
            )
            if self.itwinai_logger:
                self.itwinai_logger.log(
                    item=os.path.join(self.checkpoints_dir, "generator_weights.pth"),
                    identifier="final_generator_weights",
                    kind="artifact",
                    context="training",
                )
                self.itwinai_logger.log(
                    item=os.path.join(self.checkpoints_dir, "discriminator_weights.pth"),
                    identifier="final_discriminator_weights",
                    kind="artifact",
                    context="training",
                )

            # print("real_batch_loss", real_batch_loss)
            # print("fake_batch_loss", fake_batch_loss)
            sys.exit()

        # append mean of discriminator loss for real and fake events
        self.epoch_disc_loss.append(
            [(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)]
        )

        gen_losses[0] = [el.cpu().detach().numpy() for el in gen_losses[0]]
        gen_losses_total_loss = np.sum(gen_losses[0])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(gen_losses[0])):
            new_gen_losses.append(gen_losses[0][i_weights] / self.loss_weights[i_weights])
        gen_losses[0] = new_gen_losses

        gen_losses[1] = [el.cpu().detach().numpy() for el in gen_losses[1]]
        gen_losses_total_loss = np.sum(gen_losses[1])
        new_gen_losses = [gen_losses_total_loss]
        for i_weights in range(len(gen_losses[1])):
            new_gen_losses.append(gen_losses[1][i_weights] / self.loss_weights[i_weights])
        gen_losses[1] = new_gen_losses

        generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]

        self.epoch_gen_loss.append(generator_loss)

        # # MB: verify weight synchronization among workers
        # # Ref: https://github.com/Lightning-AI/lightning/issues/9237
        # disc_w = self.discriminator.conv1.weight.reshape(-1)[0:5]
        # gen_w = self.generator.conv1.weight.reshape(-1)[0:5]
        # print(f"DISC w: {disc_w}")
        # print(f"GEN w: {gen_w}")

        # self.index += 1 #this might be moved after test cycle

        # logging of gen and disc loss done by Trainer
        # self.log('epoch_gen_loss', self.epoch_gen_loss, on_step=True,
        #  on_epoch=True, sync_dist=True)
        # self.log('epoch_disc_loss', self.epoch_disc_loss, on_step=True, o
        # n_epoch=True, sync_dist=True)

        # return avg_disc_loss + avg_generator_loss

    def on_train_epoch_end(self):
        if not self.provenance_verbose:
            # Log provenance only at the end of an epoch
            self._log_provenance(context="training")

        discriminator_train_loss = np.mean(np.array(self.epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(self.epoch_gen_loss), axis=0)

        self.train_history["generator"].append(generator_train_loss)
        self.train_history["discriminator"].append(discriminator_train_loss)

        print("-" * 65)
        ROW_FMT = "{0:<20s} | {1:<4.2f} | {2:<10.2f} | " "{3:<10.2f}| {4:<10.2f} | {5:<10.2f}"
        print(ROW_FMT.format("generator (train)", *self.train_history["generator"][-1]))
        print(
            ROW_FMT.format("discriminator (train)", *self.train_history["discriminator"][-1])
        )

        torch.save(
            self.generator.state_dict(),
            os.path.join(self.checkpoints_dir, "generator_weights.pth"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(self.checkpoints_dir, "discriminator_weights.pth"),
        )

        if self.itwinai_logger:
            self.itwinai_logger.log(
                item=os.path.join(self.checkpoints_dir, "generator_weights.pth"),
                identifier="ckpts/generator_weights_epoch_" + str(self.current_epoch),
                kind="artifact",
                context="training",
            )
            self.itwinai_logger.log(
                item=self.generator,
                identifier="generator_epoch_" + str(self.current_epoch),
                kind="model",
                context="training",
            )
            self.itwinai_logger.log(
                item=os.path.join(self.checkpoints_dir, "discriminator_weights.pth"),
                identifier="ckpts/discriminator_weights_epoch_" + str(self.current_epoch),
                kind="artifact",
                context="training",
            )

        # with open(self.pklfile, "wb") as f:
        #     pickle.dump({"train": self.train_history,
        #                 "test": self.test_history}, f)

        # pickle.dump({"train": self.train_history}, open(self.pklfile, "wb"))
        print("train-loss:" + str(self.train_history["generator"][-1][0]))

    def validation_step(self, batch, batch_idx):
        image_batch, energy_batch, ang_batch, ecal_batch = (
            batch["X"],
            batch["Y"],
            batch["ang"],
            batch["ecal"],
        )

        image_batch = image_batch.permute(0, 4, 1, 2, 3)

        image_batch = image_batch.to(self.device)
        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)
        ecal_batch = ecal_batch.to(self.device)

        batch_size = energy_batch.shape[0]

        # Generate Fake events with same energy and angle as data batch
        noise = torch.randn(
            (batch_size, self.latent_size - 2), dtype=torch.float32, device=self.device
        )

        generator_ip = torch.cat(
            (energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise), dim=1
        )
        generated_images = self.generator(generator_ip)

        # concatenate to fake and real batches
        X = torch.cat((image_batch, generated_images), dim=0)

        # y = np.array([1] * batch_size \
        # + [0] * batch_size).astype(np.float32)
        y = torch.tensor([1] * batch_size + [0] * batch_size, dtype=torch.float32).to(
            self.device
        )
        y = y.view(-1, 1)

        ang = torch.cat((ang_batch, ang_batch), dim=0)
        ecal = torch.cat((ecal_batch, ecal_batch), dim=0)
        aux_y = torch.cat((energy_batch, energy_batch), dim=0)

        # y = [[el] for el in y]
        labels = [y, aux_y, ang, ecal]

        # Calculate discriminator loss
        disc_eval = self.discriminator(X)
        disc_eval_loss = self.compute_global_loss(labels, disc_eval, self.loss_weights)

        # Calculate generator loss
        trick = np.ones(batch_size).astype(np.float32)
        fake_batch = torch.tensor([[el] for el in trick]).to(self.device)
        # fake_batch = [[el] for el in trick]
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        generated_images = self.generator(generator_ip)
        gen_eval = self.discriminator(generated_images)
        gen_eval_loss = self.compute_global_loss(labels, gen_eval, self.loss_weights)

        if self.itwinai_logger:
            self.itwinai_logger.log(
                item=sum(disc_eval_loss),
                identifier="val_discriminator_loss",
                kind="metric",
                step=self.global_step,
                batch_idx=batch_idx,
                context="validation",
            )
            self.itwinai_logger.log(
                item=sum(gen_eval_loss),
                identifier="val_generator_loss",
                kind="metric",
                step=self.global_step,
                batch_idx=batch_idx,
                context="validation",
            )
            # Log provenance information
            if self.provenance_verbose:
                # Log provenance at every validation step
                self._log_provenance(context="validation")

        # self.log('val_discriminator_loss', sum(
        #     disc_eval_loss), on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('val_generator_loss', sum(gen_eval_loss),
        #          on_epoch=True, prog_bar=True, sync_dist=True)

        disc_test_loss = [
            disc_eval_loss[0],
            disc_eval_loss[1],
            disc_eval_loss[2],
            disc_eval_loss[3],
        ]
        gen_test_loss = [
            gen_eval_loss[0],
            gen_eval_loss[1],
            gen_eval_loss[2],
            gen_eval_loss[3],
        ]

        # Configure the loss so it is equal to the original values
        disc_eval_loss = [el.cpu().detach().numpy() for el in disc_test_loss]
        disc_eval_loss_total_loss = np.sum(disc_eval_loss)
        new_disc_eval_loss = [disc_eval_loss_total_loss]
        for i_weights in range(len(disc_eval_loss)):
            new_disc_eval_loss.append(disc_eval_loss[i_weights] / self.loss_weights[i_weights])
        disc_eval_loss = new_disc_eval_loss

        gen_eval_loss = [el.cpu().detach().numpy() for el in gen_test_loss]
        gen_eval_loss_total_loss = np.sum(gen_eval_loss)
        new_gen_eval_loss = [gen_eval_loss_total_loss]
        for i_weights in range(len(gen_eval_loss)):
            new_gen_eval_loss.append(gen_eval_loss[i_weights] / self.loss_weights[i_weights])
        gen_eval_loss = new_gen_eval_loss

        self.index += 1
        # evaluate discriminator loss
        self.disc_epoch_test_loss.append(disc_eval_loss)
        # evaluate generator loss
        self.gen_epoch_test_loss.append(gen_eval_loss)

    def _log_provenance(self, context: str):
        if self.itwinai_logger:
            # Some provenance metrics
            self.itwinai_logger.log(
                item=self.current_epoch,
                identifier="epoch",
                kind="metric",
                step=self.current_epoch,
                context=context,
            )
            self.itwinai_logger.log(
                item=self,
                identifier=f"model_version_{self.current_epoch}",
                kind="model_version",
                step=self.current_epoch,
                context=context,
            )
            self.itwinai_logger.log(
                item=None,
                identifier=None,
                kind="system",
                step=self.current_epoch,
                context=context,
            )
            self.itwinai_logger.log(
                item=None,
                identifier=None,
                kind="carbon",
                step=self.current_epoch,
                context=context,
            )
            self.itwinai_logger.log(
                item=None,
                identifier="train_epoch_time",
                kind="execution_time",
                step=self.current_epoch,
                context=context,
            )

    def on_validation_epoch_end(self):
        if not self.provenance_verbose:
            # Log provenance only at the end of an epoch
            self._log_provenance(context="validation")

        discriminator_test_loss = np.mean(np.array(self.disc_epoch_test_loss), axis=0)
        generator_test_loss = np.mean(np.array(self.gen_epoch_test_loss), axis=0)

        self.test_history["generator"].append(generator_test_loss)
        self.test_history["discriminator"].append(discriminator_test_loss)

        print("-" * 65)
        ROW_FMT = "{0:<20s} | {1:<4.2f} | {2:<10.2f} | " "{3:<10.2f}| {4:<10.2f} | {5:<10.2f}"
        print(ROW_FMT.format("generator (test)", *self.test_history["generator"][-1]))
        print(ROW_FMT.format("discriminator (test)", *self.test_history["discriminator"][-1]))

        # # save loss dict to pkl file
        # with open(self.pklfile, "wb") as f:
        #     pickle.dump({"train": self.train_history,
        #                 "test": self.test_history}, f)
        # pickle.dump({"test": self.test_history}, open(self.pklfile, "wb"))
        # print("train-loss:" + str(self.train_history["generator"][-1][0]))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        energy_batch, ang_batch = batch["Y"], batch["ang"]

        energy_batch = energy_batch.to(self.device)
        ang_batch = ang_batch.to(self.device)

        # Generate Fake events with same energy and angle as data batch
        noise = torch.randn(
            (energy_batch.shape[0], self.latent_size - 2),
            dtype=torch.float32,
            device=self.device,
        )

        # print(f"Reshape energy: {energy_batch.view(-1, 1).shape}")
        # print(f"Reshape angle: {ang_batch.view(-1, 1).shape}")
        # print(f"Noise: {noise.shape}")

        generator_ip = torch.cat(
            [energy_batch.view(-1, 1), ang_batch.view(-1, 1), noise], dim=1
        )
        # print(f"Generator input: {generator_ip.shape}")
        generated_images = self.generator(generator_ip)
        # print(f"Generated batch size {generated_images.shape}")
        return {"images": generated_images, "energies": energy_batch, "angles": ang_batch}

    def configure_optimizers(self):
        lr = self.lr

        optimizer_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr)
        optimizer_generator = torch.optim.RMSprop(self.generator.parameters(), lr)

        if self.itwinai_logger:
            self.itwinai_logger.log(
                optimizer_discriminator, "optimizer_discriminator", kind="torch"
            )
            self.itwinai_logger.log(optimizer_generator, "optimizer_generator", kind="torch")

        return [optimizer_discriminator, optimizer_generator], []
