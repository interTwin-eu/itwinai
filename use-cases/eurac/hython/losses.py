from typing import Optional, List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["RMSELoss"]

class RMSELoss(_Loss):
    def __init__(
        self,
        target_weight: dict = None,
    ):
        """
       Root Mean Squared Error (RMSE) loss for regression task.

        Parameters:
        target_weight: List of targets that contribute in the loss computation, with their associated weights.
                       In the form {target: weight}
        """
        
        super(RMSELoss, self).__init__()        
        self.mseloss = nn.MSELoss()
        self.target_weight = target_weight
        
    def forward(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.
        
        Shape
        y_true: torch.Tensor of shape (N, T).
        y_pred: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets. 
        
        Returns:
        torch.Tensor: The RMSE loss.
        """
        if self.target_weight is None:
            total_rmse_loss = torch.sqrt(self.mseloss(y_true, y_pred))
        
        else:
            total_rmse_loss = 0
            for idx, k in enumerate(self.target_weight):
                w = self.target_weight[k]
                #rmse_loss = torch.sqrt(self.mseloss(y_true[:,:,idx], y_pred[:,:,idx]))
                rmse_loss = torch.sqrt(self.mseloss(y_true[:,idx], y_pred[:,idx]))
                loss = rmse_loss * w
                total_rmse_loss += loss

        return total_rmse_loss
    
    
class MSELoss(_Loss):
    def __init__(
        self,
        weights: List
    ):
        super(RMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.weights = weights
        
    def forward(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Returns:
        torch.Tensor: The MSE loss.
        """
        mse_loss = self.mseloss(y_true, y_pred)

        return mse_loss