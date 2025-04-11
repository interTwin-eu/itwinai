
from normflow import np, torch, Model
from normflow import backward_sanitychecker
from normflow.prior import NormalPrior
from normflow.action import ScalarPhi4Action
from normflow.mask import EvenOddMask
from normflow.nn import ModuleList_, Identity_, DistConvertor_, AffineCoupling_
from normflow.nn import FFTNet_, MeanFieldNet_, PSDBlock_
from normflow.nn import ConvAct

import os

# =============================================================================
def main(kappa=0.67, m_sq=-4*0.67, lambd=0.5, n_epochs=1000, batch_size=128,
        lat_shape=(8, 8), nranks=1, **net_kwargs
        ):

    action = ScalarPhi4Action(kappa=kappa, m_sq=m_sq, lambd=lambd)

    prior = NormalPrior(shape=lat_shape)

    net_ = assemble_net(lat_shape=lat_shape, **net_kwargs)

    model = Model(net_=net_, prior=prior, action=action)

    print("number of model parameters =", model.net_.npar)

    model.net_.setup_groups(groups =
            [{'ind': [0, 1, 3], 'hyper': dict(weight_decay=1e-4)},
             {'ind': [2], 'hyper': dict(weight_decay=1e-2)}]
            )
    print(f"nranks is {nranks}")
    if nranks > 1:
        hyperparam = dict(fused=True)
    else:
        hyperparam = dict()

    snaps_dir = "../torch-snapshots"
    snaps_name= "T4_scalar_affine.E200.tar"   # if exists resume from here <name>.epoch.tar
    snaps_path = os.path.join(snaps_dir, snaps_name)
    #snaps_path = None      # set to None if you don't want to save any snapshots
    fit_kwargs = dict(
            n_epochs=n_epochs,
            save_every=200,
            batch_size=batch_size // nranks,
            hyperparam=hyperparam,
            checkpoint_dict=dict(print_stride=100, snapshot_path=snaps_path)
            )

    if nranks > 1:
        model.device_handler.spawnprocesses(fit_func, nranks, **fit_kwargs)
    else:
        model.fit(**fit_kwargs)

    backward_sanitychecker(model)
    return model


def fit_func(model, **fit_kwargs):
    model.fit(**fit_kwargs)


# =============================================================================
def assemble_net(*, lat_shape,
        n_layers=4, hidden_sizes=[8, 8], zee2sym=True, acts=None,
        knots0_len=10, knots1_len=10, knots2_len=50, knots4_len=50
        ):

    mfdict = dict(knots_len=knots0_len, symmetric=zee2sym, final_scale=True, smooth=True)

    fftdict = dict(knots_len=knots1_len, ignore_zeromode=True)

    nets_list = []
    # 1. First block
    mfnet_ = MeanFieldNet_.build(**mfdict) if (knots0_len > 1) else Identity_()
    fftnet_ = FFTNet_.build(lat_shape, **fftdict)
    nets_list.append(PSDBlock_(mfnet_=mfnet_, fftnet_=fftnet_))

    # 2. include (possible) activation
    if knots2_len > 1:
        nets_list.append(
                DistConvertor_(knots2_len, symmetric=zee2sym, smooth=True)
                    )

    # 3. Add (possible) affine blocks
    if acts is None:
       tag = 'tanh' if zee2sym else 'leaky_relu'
       acts = (*[tag]*len(hidden_sizes), None)

    conv_dict = dict(
            in_channels=1,
            out_channels=2,
            hidden_sizes=hidden_sizes,
            kernel_size=3,
            padding_mode='circular',
            conv_dim=len(lat_shape),
            acts=acts,
            bias=not zee2sym
            )
    mask = EvenOddMask(shape=lat_shape)
    nets_list.append(
            AffineCoupling_(
                    [ConvAct(**conv_dict) for _ in range(n_layers)],
                    mask=mask
            )
    )

    # 4. include (possible) activation
    if knots4_len > 1:
        nets_list.append(
                DistConvertor_(knots4_len, symmetric=zee2sym, smooth=True)
                    )

    return ModuleList_(nets_list)


# =============================================================================
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    add = parser.add_argument

    add("--lat_shape", dest="lat_shape", type=str)
    add("--m_sq", dest="m_sq", type=float)
    add("--lambd", dest="lambd", type=float)
    add("--kappa", dest="kappa", type=float)
    add("--knots0_len", dest="knots0_len", type=int)
    add("--knots1_len", dest="knots1_len", type=int)
    add("--knots2_len", dest="knots2_len", type=int)
    add("--knots4_len", dest="knots4_len", type=int)
    add("--zee2sym", dest="zee2sym", type=bool)
    add("--batch_size", dest="batch_size", type=int)
    add("--n_epochs", dest="n_epochs", type=int)
    add("--nranks", dest="nranks", type=int)
    add("--lr", dest="lr", type=float)
    add("--n_layers", dest="n_layers", type=int)
    add("--hidden_sizes", dest="hidden_sizes", type=str)

    args = vars(parser.parse_args())
    none_keys = [key for key, value in args.items() if value is None]
    [args.pop(key) for key in none_keys]
    for key in ["lat_shape", "hidden_sizes"]:
        if key in args.keys():
            args[key] = eval(args[key])

    main(**args)

