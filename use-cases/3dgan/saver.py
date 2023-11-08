from typing import Dict, Tuple, Optional
import os
import shutil

import torch
from torch import Tensor

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

        # TODO: save as 3D plot image
        for img_id, img in generated_images.items():
            img_path = os.path.join(self.save_dir, img_id + '.pth')
            torch.save(img, img_path)
