# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import os
import pickle
import random
import shutil
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from itwinai.components import Saver, monitor_exec


class ParticleImagesSaver(Saver):
    """Saves generated particle trajectories to disk."""

    def __init__(
        self, save_dir: str = "3dgan-generated", aggregate_predictions: bool = False
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))
        super().__init__()
        self.save_dir = save_dir
        self.aggregate_predictions = aggregate_predictions

    @monitor_exec
    def execute(self, generated_images: Dict[str, Tensor]) -> None:
        """Saves generated images to disk.

        Args:
            generated_images (Dict[str, Tensor]): maps unique item ID to
                the generated image.
        """
        if self.aggregate_predictions:
            os.makedirs(self.save_dir, exist_ok=True)
            sparse_generated_images = dict()
            for name, res in generated_images.items():
                sparse_generated_images[name] = res.to_sparse()
            del generated_images
            with open(self._random_file(), "wb") as fp:
                pickle.dump(sparse_generated_images, fp)
        else:
            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
            # Save as torch tensor and jpg image
            for img_id, img in generated_images.items():
                img_path = os.path.join(self.save_dir, img_id)
                torch.save(img, img_path + ".pth")
                self._save_image(img, img_id, img_path + ".jpg")

    def _random_file(self, extension: str = "pkl") -> str:
        fname = "%032x.%s" % (random.getrandbits(128), extension)
        fpath = os.path.join(self.save_dir, fname)
        while os.path.exists(fpath):
            fname = "%032x.%s" % (random.getrandbits(128), extension)
            fpath = os.path.join(self.save_dir, fname)
        return fpath

    def _save_image(
        self, img: Tensor, img_idx: str, img_path: str, center: bool = True
    ) -> None:
        """Converts a 3D tensor to a 3D scatter plot and saves it
        to disk as jpg image.
        """
        x_offset = img.shape[0] // 2 if center else 0
        y_offset = img.shape[1] // 2 if center else 0
        z_offset = img.shape[2] // 2 if center else 0

        # Convert tensor dimension IDs to coordinates
        x_coords = []
        y_coords = []
        z_coords = []
        values = []

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for z in range(img.shape[2]):
                    if img[x, y, z] > 0.0:
                        x_coords.append(x - x_offset)
                        y_coords.append(y - y_offset)
                        z_coords.append(z - z_offset)
                        values.append(img[x, y, z])

        # import plotly.graph_objects as go
        # normalize_intensity_by = 1
        # trace = go.Scatter3d(
        #     x=x_coords,
        #     y=y_coords,
        #     z=z_coords,
        #     mode='markers',
        #     marker_symbol='square',
        #     marker_color=[
        #         f"rgba(0,0,255,{i*100//normalize_intensity_by/10})"
        #         for i in values],
        # )
        # fig = go.Figure()
        # fig.add_trace(trace)
        # fig.write_image(img_path)

        values = np.array(values)
        # 0-1 scaling
        values = (values - values.min()) / (values.max() - values.min())
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x_coords, y_coords, z_coords, alpha=values)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Extract energy and angle from idx
        en, ang = img_idx.split("&")
        en = en[7:]
        ang = ang[6:]
        ax.set_title(f"Energy: {en} - Angle: {ang}")
        fig.savefig(img_path)
        plt.close()
