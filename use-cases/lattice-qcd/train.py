import sys
from normflow import Model, Fitter
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior

def make_model():
    net_ = DistConvertor_(10, symmetric=True)
    prior = NormalPrior(shape=(1,))
    action = ScalarPhi4Action(kappa=0, m_sq=-1.2, lambd=0.5)

    return Model(net_=net_, prior=prior, action=action)

def fit_func(model, n_epochs=100, strategy='ddp'):
    """Training function to fit model."""

    config = {
        "optim_lr": 0.001,
        "weight_decay": 0.01,
        "ckpt_disp": False,
        "batch_size": 128,
        "save_every": "None",
        "optimizer_class": "torch.optim.AdamW",
        "scheduler": "None",
        "loss_fn": "None",
        "print_stride": 10,
        "print_batch_size": 1024,
        "snapshot_path": None,
        "epochs_run": 0
    }
    # Initialize the Fitter and execute the training
    fitter = Fitter(model=model, epochs=n_epochs, config=config, strategy=strategy)
    fitter.execute()

def main():
    model = make_model()
    fit_func(model)

if __name__ == "__main__":
    main()
