# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""This module is used during inference to save predicted labels to file."""

import csv
import os
import shutil
from typing import Dict, List, Optional

from itwinai.components import Saver, monitor_exec


class TorchMNISTLabelSaver(Saver):
    """Serializes to disk the labels predicted for MNIST dataset."""

    def __init__(
        self,
        save_dir: str = "mnist_predictions",
        predictions_file: str = "predictions.csv",
        class_labels: Optional[List] = None,
    ) -> None:
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.save_dir = save_dir
        self.predictions_file = predictions_file
        self.class_labels = (
            class_labels if class_labels is not None else [f"Digit {i}" for i in range(10)]
        )

    @monitor_exec
    def execute(
        self,
        predicted_classes: Dict[str, int],
    ) -> Dict[str, int]:
        """Translate predictions from class idx to class label and save
        them to disk.

        Args:
            predicted_classes (Dict[str, int]): maps unique item ID to
                the predicted class ID.

        Returns:
            Dict[str, int]: predicted classes.
        """
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        # Map class idx (int) to class label (str)
        predicted_labels = {
            itm_name: self.class_labels[cls_idx]
            for itm_name, cls_idx in predicted_classes.items()
        }

        # Save to disk
        filepath = os.path.join(self.save_dir, self.predictions_file)
        with open(filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in predicted_labels.items():
                writer.writerow([key, value])
        return predicted_labels
