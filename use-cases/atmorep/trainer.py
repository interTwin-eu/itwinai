import datetime
import functools
import os
import re
import time
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import atmorep.utils.token_infos_transformations as token_infos_transformations
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data.distributed
import wandb
from atmorep.core.atmorep_model import AtmoRep, AtmoRepData
from atmorep.training.bert import prepare_batch_BERT_multifield
from atmorep.utils.utils import (
    CRPS,
    Gaussian,
    NetMode,
    init_torch,
    kernel_crps,
    setup_ddp,
    setup_wandb,
    weighted_mse,
)
from atmorep_training_configuration import AtmoRepTrainingConfiguration
from torch.distributed.optim import ZeroRedundancyOptimizer

from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import (
    TorchDDPStrategy,
)
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.type import Metric


class AtmoRepTrainer(TorchTrainer):
    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        epochs: int,
        model: Optional[nn.Module] = None,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = "ddp",
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

        # Global training configuration
        if isinstance(config, dict):
            config = AtmoRepTrainingConfiguration(**config)
            self.config = config

        if self.config.load_model is not None:
            model_id = self.config.load_model[0]
            self.config = self.config.load_json(model_id).add_backward_compatibility()

        cf = self.config
        torch.backends.cuda.matmul.allow_tf32 = True

        # calculate n_size: same for all fields
        assert "res" in cf.path_data, Exception(
            "Resolution not in file name. Please specify it."
        )
        size = np.multiply(cf.fields[0][3], cf.fields[0][4])  # ntokens x token_size
        resol = int(re.search(r"res(\d{3})", cf.path_data).group(1)) / 100
        # resol = int(cf.path_data.split("res")[1].split("_")[0]) / 100
        cf.n_size = [
            float(cf.time_sampling * size[0]),
            float(resol * size[1]),
            float(resol * size[2]),
        ]

        # transformation for token infos
        if hasattr(config, "token_infos_transformation"):
            self.tok_infos_trans = getattr(
                token_infos_transformations, cf.token_infos_transformation
            )
        else:
            self.tok_infos_trans = getattr(token_infos_transformations, "identity")

        # calculate n_size: same for all fields
        assert "res" in cf.path_data, Exception(
            "Resolution not in file name. Please specify it."
        )
        size = np.multiply(cf.fields[0][3], cf.fields[0][4])  # ntokens x token_size
        resol = int(re.search(r"res(\d{3})", cf.path_data).group(1)) / 100
        # resol = int(cf.path_data.split("res")[1].split("_")[0]) / 100
        cf.n_size = [
            float(cf.time_sampling * size[0]),
            float(resol * size[1]),
            float(resol * size[2]),
        ]

    def execute(self) -> None:
        cf = self.config

        devices = init_torch()
        self.devices = devices
        self.device_in = devices[0]
        self.device_out = devices[-1]

        cf.num_accs_per_task = len(devices)  # number of GPUs / accelerators per task
        cf.with_ddp = isinstance(self.strategy, TorchDDPStrategy)
        par_rank, par_size = setup_ddp(cf.with_ddp)
        cf.par_rank = par_rank
        cf.par_size = par_size

        # torch.cuda.set_sync_debug_mode(1)
        torch.backends.cuda.matmul.allow_tf32 = True

        # initialize random fields
        self.rng_seed = cf.rng_seed
        if not self.rng_seed:
            self.rng_seed = int(torch.randint(100000000, (1,)))
        # TODO: generate only rngs that are needed
        ll = len(cf.fields) * (max([len(f[2]) for f in cf.fields]) + 1)
        if cf.BERT_fields_synced:
            self.rngs = [np.random.default_rng(self.rng_seed) for _ in range(ll)]
        else:
            self.rngs = [np.random.default_rng(self.rng_seed + i) for i in range(ll)]

        # batch preprocessing to be done in loader (mainly for performance reasons since it's
        # parallelized there)
        self.pre_batch = functools.partial(
            prepare_batch_BERT_multifield, cf, self.rngs, cf.fields, cf.BERT_strategy
        )
        # init wandb
        self.create_logger_context()

        # create model
        # TODO: add support for pre-trained runs
        self.create_model_loss_optimizer()

        if cf.mode == "train":
            self.run()
        elif cf.mode == "evaluate":
            self.validate()
        else:
            KeyError("mode not supported. Please chose 'train' or 'validate'.")

    def create_logger_context(self):
        """
        Initialize loggers. in this case Weights and biases.
        """
        cf = self.config
        setup_wandb(cf.with_wandb, cf, cf.par_rank, project_name="train", mode="offline")

        if cf.with_wandb and 0 == cf.par_rank:
            cf.write_json(wandb)
            for key, value in cf.__dict__.items():
                print("{} : {}".format(key, value))

        if 0 == cf.par_rank:
            directory = Path(cf.path_results, "id{}".format(cf.wandb_id))

            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = Path(cf.path_models, "id{}".format(cf.wandb_id))
            if not os.path.exists(directory):
                os.makedirs(directory)

    def prepare_batch(self, xin):
        """Move data to device and some additional final preprocessing before model eval"""

        cf = self.config
        devs = self.devices

        # unpack loader output
        # xin[0] since BERT does not have targets
        (sources, token_infos, targets, fields_tokens_masked_idx_list, _) = xin[0]
        (self.sources_idxs, self.sources_info) = xin[2]

        # network input
        batch_data = [
            (
                sources[i].to(devs[cf.fields[i][1][3]], non_blocking=True),
                self.tok_infos_trans(token_infos[i]).to(self.devices[0], non_blocking=True),
            )
            for i in range(len(sources))
        ]

        # store token number since BERT selects sub-cube (optionally)
        self.num_tokens = []
        for field_idx in range(len(batch_data)):
            self.num_tokens.append(list(batch_data[field_idx][0].shape[2:5]))

        # target
        self.targets = []
        for ifield in self.fields_prediction_idx:
            self.targets.append(
                targets[ifield].to(devs[cf.fields[ifield][1][3]], non_blocking=True)
            )

        # idxs of masked tokens
        tmi_out = []
        for i, tmi in enumerate(fields_tokens_masked_idx_list):
            cdev = devs[cf.fields[i][1][3]]
            tmi_out += [[torch.cat(tmi_l).to(cdev, non_blocking=True) for tmi_l in tmi]]
        self.tokens_masked_idx = tmi_out

        return batch_data

    def encoder_to_decoder(self, embeds_layers):
        return ([embeds_layers[i][-1] for i in range(len(embeds_layers))], embeds_layers)

    def decoder_to_tail(self, idx_pred, pred):
        """Positional encoding of masked tokens for tail network evaluation"""

        cf = self.config

        field_idx = self.fields_prediction_idx[idx_pred]
        dev = self.devices[cf.fields[field_idx][1][3]]
        target_idx = self.tokens_masked_idx[field_idx]
        assert len(target_idx) > 0, "no masked tokens but target variable"

        # select "fixed" masked tokens for loss computation

        # flatten token dimensions: remove space-time separation
        pred = torch.flatten(pred, 2, 3).to(dev)
        # extract masked token level by level
        pred_masked = []
        for lidx, level in enumerate(cf.fields[field_idx][2]):
            # select masked tokens, flattened along batch dimension for easier indexing and processing
            pred_l = torch.flatten(pred[:, lidx], 0, 1)
            pred_masked.append(pred_l[target_idx[lidx]])

        # flatten along level dimension, for loss evaluation we effectively have level, batch, ...
        # as ordering of dimensions
        pred_masked = torch.cat(pred_masked, 0)

        return pred_masked

    def create(self, load_embeds=True):
        net = AtmoRep(self.config)

        # TODO: move this into a dataloader
        self.model = AtmoRepData(net)

        self.model.create(self.pre_batch, self.devices, load_embeds)

        # TODO: pass the properly to model / net
        self.model.net.encoder_to_decoder = self.encoder_to_decoder
        self.model.net.decoder_to_tail = self.decoder_to_tail
        return self

    def load(self):
        cf = self.config
        model_id, epoch = cf.load_model
        trainer = self.create(load_embeds=False)
        trainer.model.net = trainer.model.net.load(model_id, self.devices, cf, epoch)
        # TODO: pass the properly to model / net

        trainer.model.net.encoder_to_decoder = trainer.encoder_to_decoder
        trainer.model.net.decoder_to_tail = trainer.decoder_to_tail

        str = "Loaded model id = {}{}.".format(
            model_id, f" at epoch = {epoch}" if epoch > -2 else ""
        )
        print(str)
        return trainer

    ##

    def create_model_loss_optimizer(self) -> None:
        cf = self.config

        # initialize losses
        self.fields_prediction_idx = []
        self.loss_weights = torch.zeros(len(cf.fields_prediction))
        for ifield, field in enumerate(cf.fields_prediction):
            self.loss_weights[ifield] = cf.fields_prediction[ifield][1]
            for idx, field_info in enumerate(cf.fields):
                if field_info[0] == field[0]:
                    self.fields_prediction_idx.append(idx)
                    break
        self.loss_weights = self.loss_weights.to(self.device_out)
        self.MSELoss = torch.nn.MSELoss()

        # create model
        # TO-DO: fix load
        if cf.load_model is None:
            self.create()
        else:
            self.load()

        if cf.with_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, static_graph=True
            )
            if not cf.optimizer_zero:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=cf.lr_start, weight_decay=cf.weight_decay
                )
            else:
                self.optimizer = ZeroRedundancyOptimizer(
                    self.model.parameters(), optimizer_class=torch.optim.AdamW, lr=cf.lr_start
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=cf.lr_start, weight_decay=cf.weight_decay
            )

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cf.with_mixed_precision)

        if 0 == cf.par_rank:
            # print( self.model.net)
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            num_params = sum([np.prod(p.size()) for p in model_parameters])
            print(f"Number of trainable parameters: {num_params:,}")

    def get_learn_rates(self):
        cf = self.config
        size_padding = 5
        learn_rates = np.zeros(cf.num_epochs + size_padding)

        learn_rates[: cf.lr_start_epochs] = np.linspace(
            cf.lr_start, cf.lr_max, num=cf.lr_start_epochs
        )
        lr = learn_rates[cf.lr_start_epochs - 1]
        ic = 0
        for epoch in range(cf.lr_start_epochs, cf.num_epochs + size_padding):
            lr = max(lr / cf.lr_decay_rate, cf.lr_min)
            learn_rates[epoch] = lr
            if ic > 9999:  # sanity check
                assert "Maximum number of epochs exceeded."

        return learn_rates

    def run(self):
        cf = self.config
        epoch = 0 if (cf.load_model is None) else cf.load_model[1] + 1
        test_loss = np.array([1.0])
        learn_rates = self.get_learn_rates()

        # training loop
        while True:
            if epoch >= cf.num_epochs:
                break

            lr = learn_rates[epoch]
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            tstr = datetime.datetime.now().strftime("%H:%M:%S")
            print("{} : {} :: batch_size = {}, lr = {}".format(epoch, tstr, cf.batch_size, lr))

            self.train(epoch)

            if cf.with_wandb and 0 == cf.par_rank:
                self.save(epoch)

            cur_test_loss = self.validate(epoch, cf.BERT_strategy).cpu().numpy()
            # self.validate( epoch, 'forecast')

            # save model
            if cur_test_loss < test_loss.min():
                self.save(-2)
            test_loss = np.append(test_loss, [cur_test_loss])

            epoch += 1

        tstr = datetime.datetime.now().strftime("%H:%M:%S")
        print("Finished training at {} with test loss = {}.".format(tstr, test_loss[-1]))

        # save final network
        if cf.with_wandb and 0 == cf.par_rank:
            self.save(-2)

    def train(self, epoch):
        model = self.model
        cf = self.config

        model.mode(NetMode.train)
        self.optimizer.zero_grad()

        loss_total = [[] for i in range(len(cf.losses))]
        std_dev_total = [[] for i in range(len(self.fields_prediction_idx))]
        mse_loss_total = []
        grad_loss_total = []
        ctr = 0

        self.optimizer.zero_grad()
        time_start = time.time()

        for batch_idx in range(model.len(NetMode.train)):
            batch_data = self.model.next()
            _, _, _, tmksd_list, weight_list = batch_data[0]
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision
            ):
                batch_data = self.prepare_batch(batch_data)
                preds, _ = self.model(batch_data)
                loss, mse_loss, losses = self.loss(preds, weight_list)

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            [loss_total[idx].append(losses[key]) for idx, key in enumerate(losses)]
            mse_loss_total.append(mse_loss.detach().cpu())
            grad_loss_total.append(loss.detach().cpu())
            [
                std_dev_total[idx].append(pred[1].detach().cpu())
                for idx, pred in enumerate(preds)
            ]

            # logging
            if int((batch_idx * cf.batch_size) / 8) > ctr:
                # wandb logging
                if cf.with_wandb and (0 == cf.par_rank):
                    loss_dict = {
                        "training loss": torch.mean(torch.tensor(mse_loss_total)),
                        "gradient loss": torch.mean(torch.tensor(grad_loss_total)),
                    }
                # log individual loss terms for individual fields
                for idx, cur_loss in enumerate(loss_total):
                    loss_name = cf.losses[idx]
                    lt = torch.tensor(cur_loss)
                    for i, field in enumerate(cf.fields_prediction):
                        idx_name = loss_name + ", " + field[0]
                        idx_std_name = "stddev, " + field[0]
                        loss_dict[idx_name] = torch.mean(lt[:, i]).cpu().detach()
                        loss_dict[idx_std_name] = (
                            torch.mean(torch.cat(std_dev_total[i], 0)).cpu().detach()
                        )
                wandb.log(loss_dict)

                # console output
                samples_sec = cf.batch_size / (time.time() - time_start)
                str = "epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:1.5f} : {:1.5f} :: {:1.5f} ({:2.2f} s/sec)"
                print(
                    str.format(
                        epoch,
                        batch_idx,
                        model.len(NetMode.train),
                        100.0 * batch_idx / model.len(NetMode.train),
                        torch.mean(torch.tensor(grad_loss_total)),
                        torch.mean(torch.tensor(mse_loss_total)),
                        torch.mean(preds[0][1]),
                        samples_sec,
                    ),
                    flush=True,
                )

                # save model (use -2 as epoch to indicate latest, stored without
                # epoch specification)
                if batch_idx % cf.model_log_frequency == 0:
                    self.save(-2)

            # reset
            loss_total = [[] for i in range(len(cf.losses))]
            mse_loss_total = []
            grad_loss_total = []
            std_dev_total = [[] for i in range(len(self.fields_prediction_idx))]

            ctr += 1
            time_start = time.time()

        # save gradients
        if cf.save_grads and cf.with_wandb and (0 == cf.par_rank):
            dir_name = "./grads/id{}".format(cf.wandb_id)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            rmsprop_ws = []
            for k in range(len(self.optimizer.state_dict()["state"])):
                rmsprop_ws.append(
                    self.optimizer.state_dict()["state"][k]["exp_avg_sq"].mean().unsqueeze(0)
                )
            rmsprop_ws = torch.cat(rmsprop_ws)
            fname = "{}/{}_epoch{}_rmsprop.npy".format(dir_name, cf.wandb_id, epoch)
            np.save(fname, rmsprop_ws.cpu().detach().numpy())

            idx = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    fname = "{}/{}_epoch{}_{:05d}_{}_grad.npy".format(
                        dir_name, cf.wandb_id, epoch, idx, name
                    )
                    np.save(fname, param.grad.cpu().detach().numpy())
                    idx += 1

        # clean memory
        self.optimizer.zero_grad()
        del batch_data, loss, loss_total, mse_loss_total, grad_loss_total, std_dev_total

    def validate(self, epoch=0, BERT_test_strategy=None):
        cf = self.config
        if BERT_test_strategy != None:
            BERT_strategy_train = cf.BERT_strategy
            cf.BERT_strategy = BERT_test_strategy
        self.model.mode(NetMode.test)
        total_loss = 0.0
        total_losses = torch.zeros(len(self.fields_prediction_idx))
        test_len = 0

        # run test set evaluation
        with torch.no_grad():
            for it in range(self.model.len(NetMode.test)):
                batch_data = self.model.next()
                if cf.par_rank < cf.log_test_num_ranks:
                    # keep on cpu since it will otherwise clog up GPU memory
                    (sources, _, targets, tmis_list, _) = batch_data[0]
                    log_sources = (
                        [source.detach().clone().cpu() for source in sources],
                        [target.detach().clone().cpu() for target in targets],
                        tmis_list,
                    )

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=cf.with_mixed_precision
                ):
                    batch_data = self.prepare_batch(batch_data)
                    preds, atts = self.model(batch_data)
                loss = torch.tensor(0.0)
                ifield = 0
                for pred, idx in zip(preds, self.fields_prediction_idx):
                    target = self.targets[idx]
                    # base line loss
                    cur_loss = self.MSELoss(pred[0], target=target).cpu().item()

                    loss += cur_loss
                    total_losses[ifield] += cur_loss
                    ifield += 1

                total_loss += loss
                test_len += 1

                # store detailed results on current test set for book keeping
                # if cf.par_rank < cf.log_test_num_ranks :
                #     log_preds = [[p.detach().clone().cpu() for p in pred] for pred in preds]
                #     self.log_validate( epoch, it, log_sources, log_preds)
                #     if cf.attention:
                #         self.log_attention( epoch, it, atts)

        # average over all nodes
        total_loss /= test_len * len(cf.fields_prediction)
        total_losses /= test_len

        # TODO: integrate with itwinai strategy
        if cf.with_ddp:
            total_loss_cuda = total_loss.cuda()
            total_losses_cuda = total_losses.cuda()
            dist.all_reduce(total_loss_cuda, op=torch.distributed.ReduceOp.AVG)
            dist.all_reduce(total_losses_cuda, op=torch.distributed.ReduceOp.AVG)
            total_loss = total_loss_cuda.cpu()
            total_losses = total_losses_cuda.cpu()

        if 0 == cf.par_rank:
            print(
                "validation loss for strategy={} at epoch {} : {}".format(
                    BERT_test_strategy, epoch, total_loss
                ),
                flush=True,
            )
        if cf.with_wandb and (0 == cf.par_rank):
            loss_dict = {"val. loss {}".format(BERT_test_strategy): total_loss}
            total_losses = total_losses.cpu().detach()
            for i, field in enumerate(cf.fields_prediction):
                idx_name = "val., {}, ".format(BERT_test_strategy) + field[0]
                loss_dict[idx_name] = total_losses[i]
                print("validation loss for {} : {}".format(field[0], total_losses[i]))
            wandb.log(loss_dict)
        batch_data = []
        torch.cuda.empty_cache()

        if BERT_test_strategy != None:
            cf.BERT_strategy = BERT_strategy_train

        return total_loss

    def loss(self, preds, weights_list=None):
        # TODO: move implementations to individual files

        cf = self.config
        mse_loss_total = torch.tensor(
            0.0,
        )
        losses = dict(zip(cf.losses, [[] for loss in cf.losses]))

        for pred, idx in zip(preds, self.fields_prediction_idx):
            target = self.targets[idx]

            mse_loss = self.MSELoss(pred[0], target=target)
            mse_loss_total += mse_loss.cpu().detach()

            # MSE loss
            if "mse" in cf.losses:
                losses["mse"].append(mse_loss)

            # MSE loss
            if "mse_ensemble" in cf.losses:
                loss_en = torch.tensor(0.0, device=target.device)
                for en in torch.transpose(pred[2], 1, 0):
                    loss_en += self.MSELoss(en, target=target)
                losses["mse_ensemble"].append(loss_en / pred[2].shape[1])

            if "weighted_mse" in cf.losses:
                loss_en = torch.tensor(0.0, device=target.device)
                field_info = cf.fields[idx]
                token_size = field_info[4]

                weights = torch.Tensor(
                    np.array([w for batch in weights_list[idx] for w in batch])
                )
                weights = (
                    weights.view(*weights.shape, 1, 1)
                    .repeat(1, 1, token_size[0], token_size[2])
                    .swapaxes(1, 2)
                )
                weights = weights.reshape([weights.shape[0], -1]).to(target.get_device())

                for en in torch.transpose(pred[2], 1, 0):
                    loss_en += weighted_mse(en, target, weights)

                losses["weighted_mse"].append(loss_en / pred[2].shape[1])

            # Generalized cross entroy loss for continuous distributions
            if "stats" in cf.losses:
                stats_loss = Gaussian(target, pred[0], pred[1])
                diff = stats_loss - 1.0
                # stats_loss = 0.01 * torch.mean( diff * diff) + torch.mean( torch.sqrt(torch.abs( pred[1])) )
                stats_loss = torch.mean(diff * diff) + torch.mean(
                    torch.sqrt(torch.abs(pred[1]))
                )
                losses["stats"].append(stats_loss)

            # Generalized cross entroy loss for continuous distributions
            if "stats_area" in cf.losses:
                diff = torch.abs(torch.special.erf((target - pred[0]) / (pred[1] * pred[1])))
                stats_area = 0.2 * torch.mean(diff * diff) + torch.mean(
                    torch.sqrt(torch.abs(pred[1]))
                )
                losses["stats_area"].append(stats_area)

            # CRPS score
            if "crps" in cf.losses:
                crps_loss = torch.mean(CRPS(target, pred[0], pred[1]))
                losses["crps"].append(crps_loss)

            if "kernel_crps" in cf.losses:
                kcrps_loss = torch.mean(kernel_crps(target, torch.transpose(pred[2], 1, 0)))
                losses["kernel_crps"].append(kcrps_loss)

        # TODO: uncomment it and add it when running in debug mode
        # field_losses = ""
        # for ifield, field in enumerate(cf.fields):
        #   ifield_loss = 0
        #   for key in losses :
        #     ifield_loss += losses[key][ifield].to(self.device_out)
        #   ifield_loss /= len(losses.keys())
        #   field_losses +=  f"{field[0]}: {ifield_loss}; "
        # print(field_losses, flush = True)

        loss = torch.tensor(0.0, device=self.device_out)
        tot_weight = torch.tensor(0.0, device=self.device_out)
        for key in losses:
            # print( 'LOSS : {} :: {}'.format( key, losses[key]))
            for ifield, val in enumerate(losses[key]):
                loss += self.loss_weights[ifield] * val.to(self.device_out)
                tot_weight += self.loss_weights[ifield]
        loss /= tot_weight
        mse_loss = mse_loss_total / len(cf.fields_prediction)

        return loss, mse_loss, losses

    def save(self, epoch):
        self.model.net.save(epoch)
