import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=1.0, _map=False, std_init=-9):
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

    def enable_map(self):
        self.map = True

    def disenable_map(self):
        self.map = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        self.random = None
        self.map = False

    def sample_random_state(self):
        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state):
        self.random = torch.tensor(state, device=self.logsig2_w.device,
                                   dtype=self.logsig2_w.dtype)

    def KL(self):
        return 0.5 * (self.prior_prec * (self.mu_w.pow(2) + self.logsig2_w.exp())
                        - self.logsig2_w - 1 - math.log(self.prior_prec)).sum()

    def forward(self, input):
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

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"
