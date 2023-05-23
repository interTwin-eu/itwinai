"""
Base classes for pytorch lightning for itwinai.
"""

from typing import Any, List
import lightning as L


class ItwinaiBasePlModule(L.LightningModule):
    """
    Base class for pytorch lightning models for itwinai.
    """
    pass


class ItwinaiBasePlDataModule(L.LightningDataModule):
    """
    Base class for pytorch lightning models for itwinai.
    """

    def preds_to_names(
        self,
        preds: Any
    ) -> List[str]:
        """Convert predictions to class names."""
        raise NotImplementedError
