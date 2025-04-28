import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
import FrEIA.framework as ff
import FrEIA.modules as fm

from src.vblinear import VBLinear

from src.XMLHandler import XMLHandler
import h5py

from src.spline_blocks import CubicSplineBlock, RationalQuadraticSplineBlock
from src.made import MADE


class Subnet(nn.Module):
    """This class constructs a subnet for the coupling blocks"""

    def __init__(
        self,
        num_layers,
        size_in,
        size_out,
        internal_size=None,
        dropout=0.0,
        layer_class=nn.Linear,
        layer_args={},
        layer_norm=None,
        layer_act="nn.ReLU",
    ):
        """
        Initializes subnet class.

        Parameters:
        size_in: input size of the subnet
        size: output size of the subnet
        internal_size: hidden size of the subnet. If None, set to 2*size
        dropout: dropout chance of the subnet
        """
        super().__init__()
        if internal_size is None:
            internal_size = size_out * 2
        if num_layers < 1:
            raise (ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers - 1):
            if isinstance(internal_size, list):
                input_dim, output_dim = internal_size[n], internal_size[n + 1]
            else:
                input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in

            self.layer_list.append(
                layer_class[n](input_dim, output_dim, **(layer_args[n]))
            )

            if dropout > 0:
                self.layer_list.append(nn.Dropout(p=dropout))

            if layer_norm is not None:
                self.layer_list.append(eval(layer_norm)(output_dim))

            self.layer_list.append(eval(layer_act)())

        # separating last linear/VBL layer
        self.layer_list.append(
            layer_class[-1](output_dim, size_out, **(layer_args[-1]))
        )

        self.layers = nn.Sequential(*self.layer_list)

        final_layer_name = str(len(self.layers) - 1)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name and "logsig2_w" not in name:
                param.data.zero_()

    def forward(self, x):
        return self.layers(x)


class NormTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, log_cond=False):
        super().__init__(dims_in, dims_c)
        self.log_cond = log_cond

    def forward(self, x, c=None, rev=False, jac=True):
        (x,) = x
        (c,) = c
        if self.log_cond:
            c = torch.exp(c)
        if rev:
            z = x / c
            jac = -torch.log(c)
        else:
            z = x * c
            jac = torch.log(c)
        return (z,), torch.tensor([0.0], device=x.device)

    def output_dims(self, input_dims):
        return input_dims


class LogTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha=0.0, alpha_logit=0.0):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha
        self.alpha_logit = alpha_logit

    def forward(self, x, c=None, rev=False, jac=True):
        (x,) = x
        if rev:
            z = torch.exp(x) - self.alpha
            jac = torch.sum(x, dim=1)
        else:
            z = torch.log(x + self.alpha)
            jac = -torch.sum(z, dim=1)
        return (z,), torch.tensor([0.0], device=x.device)  # jac

    def output_dims(self, input_dims):
        return input_dims


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(
            dist.Uniform(torch.log(lb), torch.log(ub)), dist.ExpTransform()
        )


class CINN(nn.Module):
    """cINN model"""

    def __init__(self, data_dim, config):
        """Initializes model class.

        Parameters:
        params: Dict containing the network and training parameter
        data: Training data to initialize the norm layer
        cond: Conditions to the training data
        """
        super(CINN, self).__init__()

        self.num_dim = data_dim
        self.config = config

        self.norm_m = None
        self.use_norm = self.config.use_norm and not self.config.use_extra_dim
        self.pre_subnet = None
        self.bayesian = self.config.bayesian
        self.pre_subnet = None

        self.sub_layers = self.config.sub_layers

        if self.config.bayesian:
            self.bayesian_layers = []

        self.define_model_architecture(self.num_dim)
        print(self.model)

    def initialize_normalization(self, data, cond):
        """Calculates the normalization transformation from the training data and stores it."""
        data = torch.clone(data)
        if self.use_norm:
            data /= cond
        data = torch.log(data + self.config.alpha)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        self.norm_m = torch.diag(1 / std)
        self.norm_b = -mean / std

        print(
            "num samples out of bounds:",
            torch.count_nonzero(
                torch.max(torch.abs(data), dim=1)[0]
                > self.params.get("bounds_init", 10)
            ).item(),
        )

    def define_model_architecture(self, in_dim):
        """Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor"""

        self.in_dim = in_dim
        if self.norm_m is None:
            self.norm_m = torch.eye(in_dim)
            self.norm_b = torch.zeros(in_dim)

        nodes = [ff.InputNode(in_dim, name="inp")]
        cond_node = ff.ConditionNode(1, name="cond")

        if self.use_norm:
            nodes.append(
                ff.Node(
                    [nodes[-1].out0],
                    NormTransformation,
                    {"log_cond": self.config.log_cond},
                    conditions=cond_node,
                    name="norm",
                )
            )
        nodes.append(
            ff.Node(
                [nodes[-1].out0],
                LogTransformation,
                {"alpha": self.config.alpha, "alpha_logit": self.config.alpha_logit},
                name="inp_log",
            )
        )
        CouplingBlock, block_kwargs = self.get_coupling_block()

        for i in range(self.config.n_blocks or 10):
            if self.config.norm and i != 0:
                nodes.append(
                    ff.Node(
                        [nodes[-1].out0], fm.ActNorm, module_args={}, name=f"act_{i}"
                    )
                )
            nodes.append(
                ff.Node(
                    [nodes[-1].out0],
                    CouplingBlock,
                    block_kwargs,
                    conditions=cond_node,
                    name=f"block_{i}",
                )
            )
        nodes.append(ff.OutputNode([nodes[-1].out0], name="out"))
        nodes.append(cond_node)

        self.model = ff.GraphINN(nodes)
        self.params_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"number of parameters: {n_trainable}", flush=True)

    def get_coupling_block(self):
        """Returns the class and keyword arguments for different coupling block types"""
        constructor_fct = self.get_constructor_func()

        if self.config.coupling_type == "affine":
            CouplingBlock = fm.AllInOneBlock
            block_kwargs = {
                "affine_clamping": (
                    self.config.clamping if hasattr(self.config, "clamping") else 5
                ),
                "subnet_constructor": constructor_fct,
                "global_affine_init": 0.92,
                "permute_soft": self.config.permute_soft,
            }
        elif self.config.coupling_type == "cubic":
            CouplingBlock = CubicSplineBlock
            block_kwargs = {
                "num_bins": self.config.num_bins or 10,
                "subnet_constructor": constructor_fct,
                "bounds_init": self.config.bounds_init or 10,
                "bounds_type": self.config.bounds_type or "SOFTPLUS",
                "permute_soft": self.config.permute_soft,
            }
        elif self.config.coupling_type == "rational_quadratic":
            CouplingBlock = RationalQuadraticSplineBlock
            block_kwargs = {
                "num_bins": self.config.num_bins or 10,
                "subnet_constructor": constructor_fct,
                "bounds_init": self.config.bounds_init or 10,
                "permute_soft": self.config.permute_soft,
            }
        elif self.config.coupling_type == "MADE":
            CouplingBlock = MADE
            block_kwargs = {
                "num_bins": self.config.num_bins or 10,
                "bounds_init": self.config.bounds_init or 10,
                "permute_soft": self.config.permute_soft,
                "hidden_features": self.config.internal_size,
                "num_blocks": self.config.layers_per_block or 3,
                "dropout": self.config.dropout or 0.0,
            }
        else:
            raise ValueError(f"Unknown Coupling block type {self.config.coupling_type}")

        return CouplingBlock, block_kwargs

    def get_constructor_func(self):
        """Returns a function that constructs a subnetwork with the given parameters"""
        if self.sub_layers:
            layer_class = self.get_layer_class()
            layer_args = self.get_layer_args()
        else:
            layer_class = []
            layer_args = []
            for n in range(self.config.layers_per_block or 3):
                dicts = {}
                if self.config.bayesian:
                    layer_class.append(VBLinear)
                    dicts["prior_prec"] = self.config.prior_prec
                    dicts["std_init"] = self.config.std_init
                else:
                    layer_class.append(nn.Linear)
                layer_args.append(dicts)

        def func(x_in, x_out):
            subnet = Subnet(
                self.config.layers_per_block or 3,
                x_in,
                x_out,
                internal_size=self.config.internal_size,
                dropout=self.config.dropout or 0.0,
                layer_class=layer_class,
                layer_args=layer_args,
                layer_norm=self.config.layer_norm or None,
                layer_act=self.config.layer_act or nn.ReLU,
            )
            if self.config.bayesian:
                self.config.bayesian_layers.extend(
                    layer for layer in subnet.layer_list if isinstance(layer, VBLinear)
                )
            return subnet

        return func

    def get_layer_class(self):
        lays = []
        for n in range(len(self.sub_layers)):
            if self.sub_layers[n] == "vblinear":
                lays.append(VBLinear)
            if self.sub_layers[n] == "linear":
                lays.append(nn.Linear)
        return lays

    def get_layer_args(self):
        layer_args = []
        for n in range(len(self.sub_layers)):
            n_args = {}
            if self.sub_layers[n] == "vblinear":
                n_args["prior_prec"] = self.config.prior_prec
                n_args["std_init"] = self.config.std_init
            layer_args.append(n_args)
        return layer_args

    def forward(self, x, c, rev=False, jac=True):
        if self.config.log_cond:
            c_norm = torch.log10(
                c
            )  # use log10 for all the models (add a rescaling option)
        else:
            c_norm = c
        if self.pre_subnet:
            c_norm = self.pre_subnet(c_norm)
        return self.model.forward(x.float(), c_norm.float(), rev=rev, jac=jac)
