import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VBLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_prec: float = 1.0,
        _map: bool = False,
        std_init: float = -9.0
    ) -> None:
        """Initializes the VBLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            prior_prec (float, optional): Precision of the Gaussian prior. Defaults to 1.0.
            _map (bool, optional): Whether to operate in MAP mode. Defaults to False.
            std_init (float, optional): Initial value for log-variance parameters. Defaults to -9.0.
        """
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = _map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def enable_map(self) -> None:
        self.map = True

    def disenable_map(self) -> None:
        self.map = False

    def reset_parameters(self) -> None:
        """Initializes the parameters `mu_w`, `logsig2_w`, and `bias`."""
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self) -> None:
        self.random = None
        self.map = False

    def sample_random_state(self) -> torch.Tensor:
        """Samples a random noise tensor used for weight perturbation.

        Returns:
            torch.Tensor: A NumPy array representing the sampled random noise.
        """

        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state) -> None:
        """Imports a given random noise state for deterministic weight sampling.

        Args:
            state (np.ndarray): A NumPy array representing the random noise state.
        """
        self.random = torch.tensor(
            state, device=self.logsig2_w.device, dtype=self.logsig2_w.dtype
        )

    def KL(self) -> torch.Tensor:
        """Computes the Kullback-Leibler divergence between the variational posterior and the prior.

        Returns:
            torch.Tensor: A scalar tensor representing the KL divergence.
        """

        return (
            0.5
            * (
                self.prior_prec * (self.mu_w.pow(2) + self.logsig2_w.exp())
                - self.logsig2_w
                - 1
                - math.log(self.prior_prec)
            ).sum()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = F.linear(input, self.mu_w, self.bias)
            s2_w = self.logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:
                return F.linear(input, self.mu_w, self.bias)

            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = self.logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return F.linear(input, weight, self.bias) + 1e-8

    def __repr__(self) -> str:
        """Returns a string representation of the module."""
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"
