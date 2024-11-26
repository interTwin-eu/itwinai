
from normflow import np, torch, Model
from normflow import backward_sanitychecker
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior

import os
import sys

def fit_func(model, **fit_kwargs):
    model.fit(**fit_kwargs)

# =============================================================================
def main(
        m_sq=-1.2, lambd=0.5, knots_len=10, n_epochs=1000, batch_size=1024,
        lat_shape=1,  # basically a zero dimensional problem
        nranks=1
        ):

    net_ = DistConvertor_(knots_len, symmetric=True)

    action_dict = dict(kappa=0, m_sq=m_sq, lambd=lambd)
    prior = NormalPrior(shape=lat_shape)
    action = ScalarPhi4Action(**action_dict)

    model = Model(net_=net_, prior=prior, action=action)


    print("number of model parameters =", model.net_.npar)
    snapshot_path = "/home/csic/cdi/gsr/torch-snapshots/T4_scl0dim_test.E2000.tar"
    #snapshot_path = None

    if nranks > 1:
        hyperparam = dict(lr=0.01, weight_decay=0., fused=True)
    else:
        hyperparam = dict(lr=0.01, weight_decay=0.)

    fit_kwargs = dict(
            n_epochs=n_epochs,
            save_every=None,
            batch_size=batch_size // nranks,
            hyperparam=hyperparam,
            checkpoint_dict=dict(print_stride=100, snapshot_path=snapshot_path)
            )

    if nranks > 1:
        model.device_handler.spawnprocesses(fit_func, nranks, **fit_kwargs)
    else:
        model.fit(**fit_kwargs)

    backward_sanitychecker(model)

    return model


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--lat_shape", dest="lat_shape", type=str)
    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--kappa", dest="kappa", type=float)
    add("--knots_len", dest="knots_len", type=int)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)
    add("--nranks", dest="nranks", type=int)

    args = vars(parser.parse_args())
    none_keys = [key for key, value in args.items() if value is None]
    [args.pop(key) for key in none_keys]
    for key in ["lat_shape"]:
        if key in args.keys():
            args[key] = eval(args[key])

    main(**args)

    # print("usage: python3 scalar_model__zero_dim.py --m_sq -1.2 --lambd 0.5")
