from typing import Literal, Optional
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from operator import add

from itwinai.torch.inference import TorchModelLoader, TorchPredictor
from itwinai.loggers import Logger
from itwinai.components import monitor_exec
from itwinai.torch.config import TrainingConfiguration

from src.model import ConvVAE
from torch.utils.data import DataLoader
from src.engine import evaluate
from src.initialization import beta, criterion, n_avg, pixel_wise_criterion

class XTClimPredictor(TorchPredictor):
    def __init__(
            self,
            batch_size: int,
            evaluation: Literal["past", "future"] = 'past',
            seasons: Literal["winter_", "spring_", "summer_", "autumn_"] = 'winter_',
            strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
            model_uri: str | None = None,
            model_class: torch.nn.Module | None = None,
            logger: Logger | None = None,
            checkpoints_location: str = "checkpoints",
            name: str | None = None
    ):

        if model_class is None:
            model_class = ConvVAE

        model_loader = TorchModelLoader(model_uri, model_class) if model_uri else None
        super().__init__(
            model=model_loader,
            config={},
            strategy=strategy,
            logger=logger,
            checkpoints_location=checkpoints_location,
            name=name
        )

        self.evaluation = evaluation
        self.batch_size = batch_size
        self.seasons = seasons
        self.config = TrainingConfiguration(batch_size=batch_size)

    @monitor_exec
    def execute(self):
        # Initialize distributed backend
        self._init_distributed_strategy()

        # Define the base directory for input and output data
        input_dir = Path("input")
        output_dir = Path("outputs")

        if self.evaluation == 'past':
            season = self.seasons
            # load previously trained model
            cvae_model = self.model.to(self.device)

            distribute_kwargs = {}
            # Distributed model, optimizer, and scheduler
            cvae_model, _, _ = self.strategy.distributed(
                model=cvae_model, optimizer=None, lr_scheduler=None, **distribute_kwargs
            )

            # train set and data loader
            train_time_path = input_dir / f"dates_train_{season}data_1memb.csv"
            train_data_path = input_dir / f"preprocessed_1d_train_{season}data_1memb.npy"
            train_time = pd.read_csv(train_time_path)
            train_data = np.load(train_data_path)
            n_train = len(train_data)
            trainset = [
                (
                    torch.from_numpy(np.reshape(train_data[i], (2, 32, 32))),
                    train_time['0'][i]
                )
                for i in range(n_train)
            ]
            trainloader = self.strategy.create_dataloader(
                trainset, batch_size=self.config.batch_size, shuffle=False
            )

            # test set and data loader
            test_time_path = input_dir / f"dates_test_{season}data_1memb.csv"
            test_data_path = input_dir / f"preprocessed_1d_test_{season}data_1memb.npy"
            test_time = pd.read_csv(test_time_path)
            test_data = np.load(test_data_path)
            n_test = len(test_data)
            testset = [
                (
                    torch.from_numpy(np.reshape(test_data[i], (2, 32, 32))),
                    test_time['0'][i]
                )
                for i in range(n_test)
            ]
            testloader = self.strategy.create_dataloader(
                testset, batch_size=self.config.batch_size, shuffle=False
            )

            if self.strategy.is_main_worker and self.logger:
                self.logger.create_logger_context()

            # average over a few iterations
            # for a better reconstruction estimate
            train_avg_losses, _, tot_train_losses, _ = evaluate(
                cvae_model,
                trainloader,
                trainset,
                self.device,
                criterion,
                pixel_wise_criterion
            )
            test_avg_losses, _, tot_test_losses, _ = evaluate(
                cvae_model,
                testloader,
                testset,
                self.device,
                criterion,
                pixel_wise_criterion
            )
            for i in range(1, n_avg):
                train_avg_loss, _, train_losses, _ = evaluate(
                    cvae_model,
                    trainloader,
                    trainset,
                    self.device,
                    criterion,
                    pixel_wise_criterion
                )
                tot_train_losses = np.add(tot_train_losses, train_losses)
                train_avg_losses += train_avg_loss
                test_avg_loss, _, test_losses, _ = evaluate(
                    cvae_model,
                    testloader,
                    testset,
                    self.device,
                    criterion,
                    pixel_wise_criterion
                )
                tot_test_losses = np.add(tot_test_losses, test_losses)
                test_avg_losses += test_avg_loss
            tot_train_losses = np.array(tot_train_losses) / n_avg
            tot_test_losses = np.array(tot_test_losses) / n_avg
            train_avg_losses = train_avg_losses / n_avg
            test_avg_losses = test_avg_losses / n_avg

            pd.DataFrame(tot_train_losses).to_csv(output_dir / f"train_losses_{season}1d_allssp.csv")
            pd.DataFrame(tot_test_losses).to_csv(output_dir / f"test_losses_{season}1d_allssp.csv")
            print('Train average loss:', train_avg_losses)
            print('Test average loss:', test_avg_losses)

            self.log(train_avg_losses, 'Average train loss', kind='metric')
            self.log(test_avg_losses, 'Average test loss', kind='metric')

            # Clean-up
            if self.strategy.is_main_worker and self.logger:
                self.logger.destroy_logger_context()

            self.strategy.clean_up()

        else:
            season = self.seasons
            SCENARIO_585 = '585'
            SCENARIO_245 = '245'
            # load previously trained model
            cvae_model = ConvVAE().to(self.device)
            cvae_model.load_state_dict(self.model['model_state_dict'], strict=False)

            distribute_kwargs = {}
            # Distributed model, optimizer, and scheduler
            cvae_model, _, _ = self.strategy.distributed(
                cvae_model, **distribute_kwargs
            )

            if self.strategy.is_main_worker and self.logger:
                 self.logger.create_logger_context()

            for scenario in [SCENARIO_585, SCENARIO_245]:

                # projection set and data loader
                proj_time_path = input_dir / f"dates_proj_{season}data_1memb.csv"
                proj_data_path = input_dir / f"preprocessed_1d_proj{scenario}_{season}data_1memb.npy"
                proj_time = pd.read_csv(proj_time_path)
                proj_data = np.load(proj_data_path)
                n_proj = len(proj_data)
                projset = [
                    (
                        torch.from_numpy(np.reshape(proj_data[i], (3, 32, 32))),
                        proj_time['0'][i]
                    )
                    for i in range(n_proj)
                ]
                projloader = self.create_dataloaders(
                     projset, batch_size=1, shuffle=False
                )

                # get the losses for each data set
                # on various experiments to have representative statistics
                proj_avg_losses, _, tot_proj_losses, _ = evaluate(
                    cvae_model,
                    projloader,
                    projset,
                    self.device,
                    criterion,
                    pixel_wise_criterion
                )

                for i in range(1, n_avg):
                    proj_avg_loss, _, proj_losses, _ = evaluate(
                        cvae_model,
                        projloader,
                        projset,
                        self.device,
                        criterion,
                        pixel_wise_criterion
                    )
                    tot_proj_losses = np.add(tot_proj_losses, proj_losses)
                    proj_avg_losses += proj_avg_loss

                tot_proj_losses = np.array(tot_proj_losses) / n_avg
                proj_avg_losses = proj_avg_losses / n_avg

                # save the losses time series
                pd.DataFrame(tot_proj_losses).to_csv(output_dir / f"proj{scenario}_losses_{season}1d_allssp.csv")
                print(f'SSP{scenario} Projection average loss: {proj_avg_losses} for {season[:-1]}')

                self.log(proj_avg_losses, 'Average projection loss', kind='metric')

                # Clean-up
                if self.strategy.is_main_worker and self.logger:
                    self.logger.destroy_logger_context()

                self.strategy.clean_up()
