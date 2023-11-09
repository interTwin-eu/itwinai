from typing import Dict, Tuple, Optional
import os
import shutil

import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

from itwinai.components import Saver


class ParticleImagesSaver(Saver):
    """Saves generated particle trajectories to disk."""

    def __init__(
        self,
        save_dir: str = '3dgan-generated'
    ) -> None:
        super().__init__()
        self.save_dir = save_dir

    def execute(
        self,
        generated_images: Dict[str, Tensor],
        config: Optional[Dict] = None
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        """Saves generated images to disk.

        Args:
            generated_images (Dict[str, Tensor]): maps unique item ID to
                the generated image.
            config (Optional[Dict], optional): inherited configuration.
                Defaults to None.

        Returns:
            Tuple[Optional[Tuple], Optional[Dict]]: propagation of inherited
                configuration and saver return value.
        """
        result = self.save(generated_images)
        return ((result,), config)

    def save(self, generated_images: Dict[str, Tensor]) -> None:
        """Saves generated images to disk.

        Args:
            generated_images (Dict[str, Tensor]): maps unique item ID to
                the generated image.
        """
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        # Save as torch tensor and jpg image
        for img_id, img in generated_images.items():
            img_path = os.path.join(self.save_dir, img_id)
            torch.save(img, img_path + '.pth')
            self._save_image(img, img_id, img_path + '.jpg')

    def _save_image(
        self,
        img: Tensor,
        img_idx: str,
        img_path: str,
        center: bool = True
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
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, alpha=values)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Extract energy and angle from idx
        en, ang = img_idx.split('&')
        en = en[7:]
        ang = ang[6:]
        ax.set_title(f"Energy: {en} - Angle: {ang}")
        fig.savefig(img_path)
