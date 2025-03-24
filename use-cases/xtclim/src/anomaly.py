from typing import Any, Dict, Literal, Optional
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
        self.config = TrainingConfiguration(batch_size=batch_size, shuffle_test=False)


    def read_data_and_dataloader(
        self,
        input_dir: Path,
        season: str,
        data_type: str
    ) -> tuple[Dataset, int]:
        """Loads dataset from specified directory and returns the dataloader.

        Args:
            input_dir (Path): The path to the directory containing the dataset.
            season (str): The season identifier used for data filenames.
            data_type (str): The type of dataset to load ('train', 'test', 'proj').

        Returns:
            tuple[Dataset, int]: A tuple with the dataset (Dataset) and its length (int).
        """
        time_path = input_dir / f"dates_{data_type}_{season}data_1memb.csv"
        data_path = input_dir / f"preprocessed_1d_{data_type}_{season}data_1memb.npy"

        time_data = pd.read_csv(time_path)
        data = np.load(data_path)

        n_data = len(data)
        dataset = [
            (
                torch.from_numpy(np.reshape(data[i], (2, 32, 32))),
                time_data['0'][i]
            )
            for i in range(n_data)
        ]

        return dataset, n_data

    def evaluate_and_average(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset: Dataset,
        device: str,
        criterion: torch.nn.Module,
        pixel_wise_criterion: torch.nn.Module,
        n_avg: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates model on dataset and averages the losses over `n_avg` iterations.

        Args:
            model (nn.Module): The trained model.
            dataloader (DataLoader): The DataLoader for the dataset.
            dataset (Dataset): The dataset to evaluate the model on.
            device (str): Device (e.g., 'cuda', 'cpu').
            criterion (nn.Module): Loss function.
            pixel_wise_criterion (nn.Module): Additional loss function for pixel-wise evaluation.
            n_avg (int): Number of iterations to average over.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing averaged losses for the dataset.
        """
        avg_losses = 0
        total_losses = []

        for i in range(n_avg):
            avg_loss, _, losses, _ = evaluate(
                model, dataloader, dataset, device, criterion, pixel_wise_criterion
            )
            avg_losses += avg_loss
            total_losses.append(losses)

        total_losses = np.mean(np.stack(total_losses), axis=0)
        avg_losses = avg_losses / n_avg

        return avg_losses, total_losses

    def predict(self) -> Dict[str, Any]:
        """Predict to use evaluate_and_average instead of direct model inference.

        Returns:
            Dict[str, Any]: A dictionary containing average and total losses.
        """

        avg_losses, total_losses = self.evaluate_and_average(
            model=self.model,
            dataloader=self.inference_dataloader,
            dataset=self.inference_dataset,
            device=self.device,
            criterion=criterion,
            pixel_wise_criterion=pixel_wise_criterion,
            n_avg=n_avg
        )

        predictions = {
            "avg_losses": float(avg_losses),
            "total_losses": total_losses.tolist() if isinstance(total_losses, np.ndarray) else total_losses
        }

        return predictions


    def save_to_csv(self, losses: list | np.ndarray, output_dir: Path, filename: str) -> None:
        """Saves to CSV file at specified location.

        Args:
            losses (list | np.ndarray): Losses to be saved.
            output_dir (Path): Directory to save CSV file.
            filename (str): Name of the CSV file.

        Returns:
            None
        """
        pd.DataFrame(losses).to_csv(output_dir / filename)


    @monitor_exec
    def execute(self):

        # Define the base directory for input and output data
        input_dir = Path("input")
        output_dir = Path("outputs")

        if isinstance(self.model, TorchModelLoader):
            self.model = self.model()
        else:
            raise ValueError("Error: model is not an instance of TorchModelLoader.")

        if self.evaluation == 'past':

            trainset, n_train = self.read_data_and_dataloader(input_dir, self.seasons, "train")
            self.inference_dataset = trainset
            train_predictions = super().execute(inference_dataset=self.inference_dataset, model=self.model)

            testset, n_test = self.read_data_and_dataloader(input_dir, self.seasons, "test")
            self.inference_dataset = testset
            test_predictions = super().execute(inference_dataset=self.inference_dataset, model=self.model)

            train_avg_losses = train_predictions["avg_losses"]
            tot_train_losses = train_predictions["total_losses"]
            test_avg_losses = test_predictions["avg_losses"]
            tot_test_losses = test_predictions["total_losses"]

            # Export to CSV
            self.save_to_csv(tot_train_losses, output_dir, f"train_losses_{self.seasons}1d_allssp.csv")
            self.save_to_csv(tot_test_losses, output_dir, f"test_losses_{self.seasons}1d_allssp.csv")

            # Logging
            print('Train average loss:', train_avg_losses)
            print('Test average loss:', test_avg_losses)
            self.log(train_avg_losses, 'Average train loss', kind='metric')
            self.log(test_avg_losses, 'Average test loss', kind='metric')

        else:
            SCENARIO_585 = '585'
            SCENARIO_245 = '245'

            for scenario in [SCENARIO_585, SCENARIO_245]:

                # projection set and data loader
                projset, n_proj = read_data_and_dataloader(input_dir, self.seasons, "proj{scenario}")

                self.config.batch_size = 1

                self.inference_dataset = projset
                proj_predictions = super().execute(inference_dataset=self.inference_dataset, model=self.model)

                proj_avg_losses = proj_predictions["avg_losses"]
                tot_proj_losses = proj_predictions["total_losses"]

                # Export to CSV
                self.save_to_csv(
                    tot_proj_losses,
                    output_dir,
                    f"proj{scenario}_losses_{self.seasons}1d_allssp.csv"
                )
                print(f'SSP{scenario} Projection average loss: {proj_avg_losses} for {self.seasons[:-1]}')

                # Logging
                self.log(proj_avg_losses, 'Average projection loss', kind='metric')
