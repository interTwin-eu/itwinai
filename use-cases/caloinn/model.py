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

from typing import Any, Dict, Tuple, Callable, List, Type


class Subnet(nn.Module):
    """Constructs a subnet for coupling blocks."""

    def __init__(
        self,
        num_layers: int,
        size_in: int,
        size_out: int,
        internal_size: int = None,
        dropout: float = 0.0,
        layer_class=nn.Linear,
        layer_args={},
        layer_norm: str = None,
        layer_act: str = "nn.ReLU",
    ):
        """Initializes the Subnet module.

        Args:
            num_layers (int): The number of layers in the subnet, excluding the output layer. 
                Must be at least 1. Each layer is followed by an activation function, dropout, 
                and possibly a normalization layer.
            size_in (int): The number of input features to the subnet (dimension of the input tensor).
            size_out (int): The number of output features from the subnet (dimension of the output tensor).
            internal_size (int or list of int, optional): The size(s) of the hidden layers. 
                If None, defaults to 2 * size_out. If a list is provided, it defines the size for each hidden layer.
            dropout (float, optional): The dropout probability applied after each hidden layer to reduce overfitting. 
                A value between 0.0 and 1.0. Defaults to 0.0 (no dropout).
            layer_class (type or list of types, optional): The class used for the layers in the subnet, typically 
                nn.Linear or a similar module like VBLinear. If a list is provided, each layer will have a 
                corresponding class from the list. Defaults to nn.Linear.
            layer_args (list of dicts, optional): A list of dictionaries containing arguments to pass 
                to each layer constructor. Each dictionary should match the parameters expected by the 
                `layer_class`. The length of the list should correspond to the number of layers in the subnet.
            layer_norm (str, optional): The name of the normalization layer class to use after each hidden layer.
                For example, 'nn.BatchNorm1d'. Defaults to None (no normalization).
            layer_act (str, optional): The activation function to use for hidden layers. Should be a string
                corresponding to an activation function class, such as 'nn.ReLU' or 'nn.LeakyReLU'.
                Defaults to 'nn.ReLU'.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the subnet to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to the subnet. It is expected to have shape 
                (batch_size, size_in), where `size_in` is the number of input features.

        Returns:
            torch.Tensor: The output tensor after passing through the subnet. It will have shape 
                (batch_size, size_out), where `size_out` is the number of output features.
        """
        return self.layers(x)


class NormTransformation(fm.InvertibleModule):
    def __init__(self, dims_in: tuple, dims_c: tuple = None, log_cond: bool = False):
        """Initializes the NormTransformation module.

        Args:
            dims_in (tuple): Input dimensions for the transformation (e.g., (batch_size, num_features)).
            dims_c (tuple, optional): Condition dimensions for the transformation. 
                It can be None or a tensor of matching dimensions for the condition. Defaults to None.
            log_cond (bool, optional): If True, exponentiates the condition `c`. Defaults to False.
        """
        super().__init__(dims_in, dims_c)
        self.log_cond = log_cond

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, rev: bool = False, jac: bool = True) -> tuple:
        """Applies the forward or inverse transformation.

        Args:
            x (tuple of torch.Tensor): Input tensor(s), with `x` being the first element.
            c (tuple of torch.Tensor, optional): Condition tensor(s), with `c` being the first element. 
                If log_cond is True, the condition will be exponentiated before use.
            rev (bool, optional): If True, applies the inverse transformation. Defaults to False (forward transformation).
            jac (bool, optional): If True, returns the Jacobian term (though not used in this case). Defaults to True.

        Returns:
            tuple: A tuple containing the transformed tensor(s) and a dummy Jacobian tensor (as a tensor of zeros).
        """
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

    def output_dims(self, input_dims: tuple) -> tuple:
        """Returns the output dimensions, which are the same as the input dimensions.

        Args:
            input_dims (tuple): Input dimensions.

        Returns:
            tuple: The output dimensions, which are the same as the input dimensions.
        """
        return input_dims


class LogTransformation(fm.InvertibleModule):
    """Applies a logarithmic transformation with an additive shift."""
    
    def __init__(self, dims_in: tuple, dims_c: tuple = None, alpha: float = 0.0, alpha_logit: float = 0.0):
        """Initializes the LogTransformation module.

        Args:
            dims_in (tuple): Input dimensions for the transformation (e.g., (batch_size, num_features)).
            dims_c (tuple, optional): Condition dimensions for the transformation. It is not used 
                in this class, so it can be None. Defaults to None.
            alpha (float, optional): A small value to be added before taking the logarithm to avoid log(0).
                Defaults to 0.0.
            alpha_logit (float, optional): Placeholder for future functionality. Defaults to 0.0.
        """
        super().__init__(dims_in, dims_c)
        self.alpha = alpha
        self.alpha_logit = alpha_logit

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, rev: bool = False, jac: bool = True) -> tuple:
        """Applies the forward or inverse log transformation.

        Args:
            x (tuple of torch.Tensor): Input tensor(s).
            c (tuple of torch.Tensor, optional): Condition tensor(s). Not used in this function.
            rev (bool, optional): If True, applies the inverse transformation (exponentiation). Defaults to False.
            jac (bool, optional): If True, returns the Jacobian term (though not used here). Defaults to True.

        Returns:
            tuple: A tuple containing the transformed tensor(s) and a dummy Jacobian tensor (as a tensor of zeros).
        """
        (x,) = x
        if rev:
            z = torch.exp(x) - self.alpha
            jac = torch.sum(x, dim=1)
        else:
            z = torch.log(x + self.alpha)
            jac = -torch.sum(z, dim=1)
        return (z,), torch.tensor([0.0], device=x.device)  # jac

    def output_dims(self, input_dims: tuple) -> tuple:
        """Returns the output dimensions, which are the same as the input dimensions.

        Args:
            input_dims (tuple): Input dimensions.

        Returns:
            tuple: The output dimensions, which are the same as the input dimensions.
        """
        return input_dims


class LogUniform(dist.TransformedDistribution):
    """Defines a log-uniform distribution between two bounds."""

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        """Initializes the LogUniform distribution.

        Args:
            lb (torch.Tensor): Lower bound for the distribution. The logarithm of this value is used.
            ub (torch.Tensor): Upper bound for the distribution. The logarithm of this value is used.
        """
        super(LogUniform, self).__init__(
            dist.Uniform(torch.log(lb), torch.log(ub)), dist.ExpTransform()
        )


class CINN(nn.Module):
    """Conditional Invertible Neural Network (cINN) model."""

    def __init__(self, data_dim: int, config: Any) -> None:
        """Initializes the CINN model.

        Args:
            data_dim (int): Dimensionality of the input data.
            config (Any): Configuration object containing model and training parameters.
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

    def initialize_normalization(self, data: torch.Tensor, cond: torch.Tensor) -> None:
        """Calculates and stores normalization transformations based on training data.

        Args:
            data (torch.Tensor): Input data tensor.
            cond (torch.Tensor): Condition tensor associated with the input data.
        """
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

    def define_model_architecture(self, in_dim: int) -> None:
        """Creates a reversible network model based on the provided configuration.

        Args:
            in_dim (int): Dimensionality of the input space.
        """

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

    def get_coupling_block(self) -> Tuple[Any, Dict[str, Any]]:
        """Retrieves the coupling block class and corresponding keyword arguments.

        Returns:
            Tuple containing the coupling block class and a dictionary of keyword arguments.
        """
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

    def get_constructor_func(self) -> Callable:
        """Returns a function to construct subnetwork layers.

        Returns:
            Callable that builds a subnetwork given input and output dimensions.
        """
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

    def get_layer_class(self) -> List[Type[nn.Module]]:
        """Retrieves the list of layer classes based on configuration.

        Returns:
            List of layer classes.
        """
        lays = []
        for n in range(len(self.sub_layers)):
            if self.sub_layers[n] == "vblinear":
                lays.append(VBLinear)
            if self.sub_layers[n] == "linear":
                lays.append(nn.Linear)
        return lays

    def get_layer_args(self) -> List[Dict[str, Any]]:
        """Retrieves the list of argument dictionaries for each layer.

        Returns:
            List of dictionaries with layer-specific arguments.
        """
        layer_args = []
        for n in range(len(self.sub_layers)):
            n_args = {}
            if self.sub_layers[n] == "vblinear":
                n_args["prior_prec"] = self.config.prior_prec
                n_args["std_init"] = self.config.std_init
            layer_args.append(n_args)
        return layer_args

    def forward(self, x: torch.Tensor, c: torch.Tensor, rev: bool = False, jac: bool = True) -> Any:
        """Forward or inverse pass through the network.

        Args:
            x (torch.Tensor): Input data.
            c (torch.Tensor): Conditioning data.
            rev (bool, optional): Whether to perform the inverse transformation. Defaults to False.
            jac (bool, optional): Whether to compute the Jacobian. Defaults to True.

        Returns:
            Output of the model forward pass.
        """
        if self.config.log_cond:
            c_norm = torch.log10(
                c
            )  # use log10 for all the models (add a rescaling option)
        else:
            c_norm = c
        if self.pre_subnet:
            c_norm = self.pre_subnet(c_norm)
        return self.model.forward(x.float(), c_norm.float(), rev=rev, jac=jac)
