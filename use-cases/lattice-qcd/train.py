import sys
from normflow import Model, Fitter
from normflow.nn import DistConvertor_
from normflow.action import ScalarPhi4Action
from normflow.prior import NormalPrior

from itwinai.loggers import ConsoleLogger, MLFlowLogger, LoggersCollection

def make_model():
    net_ = DistConvertor_(10, symmetric=True)
    prior = NormalPrior(shape=(1,))
    action = ScalarPhi4Action(kappa=0, m_sq=-1.2, lambd=0.5)

    model = Model(net_=net_, prior=prior, action=action)

    return model

def fit_func(
        model,
        n_epochs=100,
        batch_size=128,
    ):
    """Training function to fit model."""

    # Initialize the Fitter and execute the training
    fitter = Fitter(
        model=model,
        epochs=n_epochs,  # Pass number of epochs
    )

    fitter.execute(
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

def main():
    model = make_model()
    fit_func(model)

if __name__ == "__main__":
    main()
    sys.exit()
