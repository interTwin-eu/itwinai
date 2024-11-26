# Copyright (c) 2021-2022 Javad Komijani


import torch
import numpy as np
import copy
import io
import base64


# =============================================================================
class Module_(torch.nn.Module):
    """A prototype class: like a `torch.nn.Module` except for the `forward`
    and `backward` methods that handle the Jacobians of the transformation.
    We use trailing underscore to denote the neworks in which the `forward`
    and `backward` methods handle the Jacobians of the transformation.
    """

    propagate_density = False

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    def forward(self, x, log0=0):
        pass

    def backward(self, x, log0=0):
        pass

    def transfer(self, **kwargs):
        return copy.deepcopy(self)

    @property
    def npar(self):
        return sum([np.prod(p.shape) for p in self.parameters()])

    def sum_density(self, x):
        if self.propagate_density:
            return x
        else:
            return torch.sum(x, dim=list(range(1, x.dim())))


# =============================================================================
class ModuleList_(torch.nn.ModuleList):
    """Like `torch.nn.ModuleList` except for the `forward` and `backward`
    methods that handle the Jacobians of the transformation.
    We use trailing underscore to denote the neworks in which the `forward`
    and `backward` methods handle the Jacobians of the transformation.

    Parameters
    ----------
    nets_ : list
        Items of the list must be instances of Module_ or ModuleList_
    """

    _groups = None

    def __init__(self, nets_, label=None):
        super().__init__(nets_)
        self.label = label

    def forward(self, x, log0=0):
        for net_ in self:
            x, log0 = net_.forward(x, log0)
        return x, log0

    def backward(self, x, log0=0):
        for net_ in list(self)[::-1]:  # list() is needed for child classes...
            x, log0 = net_.backward(x, log0)
        return x, log0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def grouped_parameters(self):
        if self._groups is None:
            return super().parameters()
        else:
            params_list = []
            sum_ = lambda x: sum(x, start=[])
            for grp in self._groups:
                par = sum_([list(self[k].parameters()) for k in grp['ind']])
                params_list.append(dict(params=par, **grp['hyper']))
            return params_list

    def setup_groups(self, groups=None):
        """If group is not None, it must be a list of dicts. e.g. as
        groups = [{'ind': [0, 1], 'hyper': dict(weight_decay=1e-4)},
                  {'ind': [2, 3], 'hyper': dict(weight_decay=1e-2)}]
        """
        self._groups = groups

    def hack(self, x, log0=0):
        """Similar to the forward method, except that returns the output of
        middle blocks too; useful for examining effects of each block.
        """
        stack = [(x, log0)]
        for net_ in self:
            x, log0 = net_.forward(x, log0)
            stack.append((x, log0))
        return stack

    def transfer(self, **kwargs):
        return self.__class__([net_.transfer(**kwargs) for net_ in self])

    def get_weights_blob(self):
        serialized_model = io.BytesIO()
        torch.save(self.state_dict(), serialized_model)
        return base64.b64encode(serialized_model.getbuffer()).decode('utf-8')

    def set_weights_blob(self, blob, map_location=torch.device('cpu')):
        weights = torch.load(
                io.BytesIO(base64.b64decode(blob.strip())),
                map_location=map_location
                )
        self.load_state_dict(weights)

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def npar(self):
        return sum([np.prod(p.shape) for p in super().parameters()])

    def to(self, *args, **kwargs):
        for net_ in self:
            net_.to(*args, **kwargs)


# =============================================================================
class MultiChannelModule_(torch.nn.ModuleList):
    """A prototype class similar to `Module_` except that it handles multiple
    channels seperately, in the sense that each channel is transformed by
    corresponding NN. The number of input NNs must agree with the number of
    channels.
    """

    def __init__(self, nets_,
            label=None, channels_axis=1, keep_channels_axis=True):
        super().__init__(nets_)
        self.channels_axis = channels_axis
        self.keep_channels_axis = keep_channels_axis
        self.label = label

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, log0=0):
        return self._map(x, [net_.forward for net_ in self], log0=log0)

    def backward(self, x, log0=0):
        return self._map(x, [net_.backward for net_ in self], log0=log0)

    def _map(self, x, f_, log0=0):
        if self.keep_channels_axis:
            x = x.split(1, dim=self.channels_axis)
        else:
            x = x.unbind(dim=self.channels_axis)

        assert len(x) == len(f_), "mismatch in channels of input & network."

        out = [fj_(xj) for fj_, xj in zip(f_, x)]
        if self.keep_channels_axis:
            x = torch.cat([o[0] for o in out], dim=self.channels_axis)
        else:
            x = torch.stack([o[0] for o in out], dim=self.channels_axis)
        logJ = sum([o[1] for o in out])

        return x, log0 + logJ

    def parameters(self):
        return super().parameters()

    @property
    def npar(self):
        return sum([np.prod(p.shape) for p in super().parameters()])


# =============================================================================
class MultiOutChannelModule_(MultiChannelModule_):

    def _map(self, x, f_, log0=0):

        out = [fj_(x) for fj_ in f_]
        x = torch.cat([o[0] for o in out], dim=self.channels_axis)
        logJ = sum([o[1] for o in out])

        return x, log0 + logJ


# =============================================================================
class InvisibilityMaskWrapperModule_(Module_):
    """A wrapper that makes a part of the input invisible before passing it the
    underlying network (`net_`). 

    Parameters
    ----------
    net_ : instance of Module_
        should not have any other nested net_ that keeps track of Jacobian
        of transformation.

    mask : instance of Mask
        for partitioning the input data to visible and invisible parts.
    """

    def __init__(self, net_, *, mask):
        super().__init__(label=f'wrapper:{net_.label}')
        self.net_ = net_
        self.mask = mask
        self.net_.propagate_density = True  # does not sum the density

    def forward(self, x, log0=0):
        x_v, x_invisible = self.mask.split(x)  # x_v: x_visible
        x_v, logJ_density = self.net_.forward(x_v)
        x_v = self.mask.purify(x_v, channel=0)
        logJ = self.sum_density(self.mask.purify(logJ_density, channel=0))
        return self.mask.cat(x_v, x_invisible), log0 + logJ

    def backward(self, x, log0=0):
        x_v, x_invisible = self.mask.split(x)  # x_v: x_visible
        x_v, logJ_density = self.net_.backward(x_v)
        x_v = self.mask.purify(x_v, channel=0)
        logJ = self.sum_density(self.mask.purify(logJ_density, channel=0))
        return self.mask.cat(x_v, x_invisible), log0 + logJ
