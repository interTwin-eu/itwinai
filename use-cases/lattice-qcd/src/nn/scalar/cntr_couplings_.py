# Copyright (c) 2023 Javad Komijani

"""
This module contains special cases of coupling layers that are controled.

As in Module_, the trailing underscore implies that the associated forward and
backward methods handle the Jacobians of the transformation.
"""


from .couplings_ import Coupling_
from .couplings_ import ShiftCoupling_, AffineCoupling_
from .couplings_ import RQSplineCoupling_, MultiRQSplineCoupling_


# =============================================================================
class DirectCntrCoupling_(Coupling_):
    """A "controlled" version of Coupling_."""

    def forward(self, x_and_control, log0=0):
        x, control = x_and_control
        x = list(self.mask.split(x))  # x = [x_0, x_1]
        for k, net in enumerate(self.nets):
            parity = k % 2
            x_frozen = control if k == 0 else x[1 - parity]
            x[parity], log0 = self.atomic_forward(
                                                  x_active=x[parity],
                                                  x_frozen=x_frozen,
                                                  parity=parity,
                                                  net=net,
                                                  log0=log0
                                                  )
        x_and_control = (self.mask.cat(*x), control)
        return x_and_control, log0

    def backward(self, x_and_control, log0=0):
        x, control = x_and_control
        x = list(self.mask.split(x))  # x = [x_0, x_1]
        for k in list(range(len(self.nets)))[::-1]:
            parity = k % 2
            x_frozen = control if k == 0 else x[1 - parity]
            x[parity], log0 = self.atomic_backward(
                                                  x_active=x[parity],
                                                  x_frozen=x_frozen,
                                                  parity=parity,
                                                  net=self.nets[k],
                                                  log0=log0
                                                  )
        x_and_control = (self.mask.cat(*x), control)
        return x_and_control, log0


# =============================================================================
class CntrCoupling_(DirectCntrCoupling_):
    """Similar to DirectCntrCoupling_ except that it is not direct; the
    user will not see the control term.

    This class accepts a generator at the time of instantiating.
    In each call, a control term will be generated for controling the output,
    but the will not be returned; the control term will be saved for a
    reference, but will be replaced in the next call.
    """

    def __init__(self, *args, control_generator=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.control_generator = control_generator
        self.control = None

    def forward(self, x, log0=0):
        batch_size = x.shape[0]
        self.control = self.control_generator(batch_size)
        (x, control), log0 = super().forward((x, self.control), log0=log0)
        return x, log0

    def backward(self, x, log0=0):
        (x, control), log0 = super().backward((x, self.control), log0=log0)
        return x, log0


# =============================================================================
class CntrShiftCoupling_(CntrCoupling_, ShiftCoupling_):
    pass


class CntrAffineCoupling_(CntrCoupling_, AffineCoupling_):
    pass


class CntrRQSplineCoupling_(CntrCoupling_, RQSplineCoupling_):
    pass


class CntrMultiRQSplineCoupling_(CntrCoupling_, MultiRQSplineCoupling_):
    pass
