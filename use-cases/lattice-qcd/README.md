normflow
[![SQAaaS badge shields.io](https://img.shields.io/badge/sqaaas%20software-silver-lightgrey)](https://api.eu.badgr.io/public/assertions/-g9rQYZJTyi4S-VUrbvqlQ "SQAaaS silver badge achieved")
[![License: MIT](https://cdn.prod.website-files.com/5e0f1144930a8bc8aace526c/65dd9eb5aaca434fac4f1c34_License-MIT-blue.svg)](LICENSE)
--------
This package provides utilities for implementing the
**method of normalizing flows** as a generative model for lattice field theory.

The method of normalizing flows is a powerful generative modeling approach that
learns complex probability distributions by transforming samples from a simple
distribution through a series of invertible transformations. It has found
applications in various domains, including generative image modeling.

The package currently supports scalar theories in any dimension, and we are
actively extending it to accommodate gauge theories, broadening its
applicability.

In a nutshell, three essential components are required for the method of
normalizing flows:

*   A **prior distribution** to draw initial samples.
*   A **neural network** to perform a series of invertible transformations on
    the samples.
*   An **action** that specifies the target distribution, defining the goal of
    the generative model.

The central high-level class of the package is called `Model`, which can be
instantiated by providing instances of the three objects mentioned above:
the prior, the neural network, and the action.

The `Fitter` class trains this instance of the `Model` class, which is
instantiated during the initialization of the `Fitter` class. The `Fitter`
class inherits from the `TorchTrainer` class provided by `itwinai`. Importantly,
the distributed training logic is obtained through this class and allows the
user to seamlessly integrate any one of the multiple distributed training
strategies configured in `itwinai`. Besides, it also provides functionalities
to perform profiling, logging and HyperParameter Optimization (HPO), which
are all inherited from the TorchTrainer class.

The training configuration is handled with `yaml` based configuration files.
An example of such a configuration file is provided in `config.yaml`. An example
to demonstrate the scalar theory in zero dimension, i.e., a scenario with one
point and one degree of freedom is provided in this file.

```yaml
model:
    _target_: normflow.Model
    net_:
        _target_: normflow.nn.DistConvertor_
        knots_len: 10
        symmetric: True
    prior:
        _target_: normflow.prior.NormalPrior
        shape: [1]
    action:
        _target_: normflow.action.ScalarPhi4Action
        kappa: 0
        m_sq: -1.2
        lambd: 0.5
```

In this example, we have:

-   **Prior Distribution**: A normal distribution is used with a shape of`[1]`.

-   **Action**: A quartic scalar theory is defined with parameters
    `kappa=0`, `m_sq=-2.0`, and `lambda=0.2`.

-   **Neural Network**: The `DistConvertor_` class is used to create the
    transformation network, with `knots_len=10` and symmetry enabled.
    Any instance of this class converts the probability distribution of inputs
    using a rational quadratic spline. In this example, the spline has 10 knots,
    and the distribution is assumed to be symmetric with respect to the origin.

The other parameters for training are specified within the same `yaml` file.
For conciseness, only few of the options, the `config` parameter, epochs and strategy
are shown in the snippet here.
```yaml
config:
    optim_lr: 0.001
    weight_decay: 0.01
    batch_size: 128
epochs: 100
strategy: "ddp"
```
These settings will train the model for `100` epochs with a batch size of `128`,
learning rate `optim_lr` of `0.001` and `weight_decay` of `0.01`. The strategy
for distributing the training is specified with the `strategy` flag, which is set
to `ddp` here, which implies the default PyTorch-based Distributed Data Parallel (DDP)
strategy.

In order to launch the pipeline, simply run:
```
itwinai exec-pipeline --config-name config.yaml +pipe_key=training_pipeline
```
The above code block results in an output similar to:

```
>>> Training progress (cpu) <<<

Note: log(q/p) is estimated with normalized p; mean & error are obtained from samples in a batch
Epoch: 1 | loss: -0.5387 | ess: 0.8755
ConsoleLogger: epoch_loss = -0.5386953949928284
Epoch: 10 | loss: -0.4528 | ess: 0.8750
ConsoleLogger: epoch_loss = -0.45278581976890564
Epoch: 20 | loss: -0.7644 | ess: 0.8989
ConsoleLogger: epoch_loss = -0.76436448097229
Epoch: 30 | loss: -0.6655 | ess: 0.8778
ConsoleLogger: epoch_loss = -0.6654551029205322
Epoch: 40 | loss: -0.7596 | ess: 0.8895
ConsoleLogger: epoch_loss = -0.7596336603164673
Epoch: 50 | loss: -0.7449 | ess: 0.8836
ConsoleLogger: epoch_loss = -0.7449398040771484
Epoch: 60 | loss: -0.7312 | ess: 0.8920
ConsoleLogger: epoch_loss = -0.7311607599258423
Epoch: 70 | loss: -0.8115 | ess: 0.8982
ConsoleLogger: epoch_loss = -0.8114994764328003
Epoch: 80 | loss: -0.8218 | ess: 0.9046
ConsoleLogger: epoch_loss = -0.8217660188674927
Epoch: 90 | loss: -0.7733 | ess: 0.9065
ConsoleLogger: epoch_loss = -0.7732580900192261
Epoch 100 | Model Snapshot saved at checkpoint.E100.tar
Epoch: 100 | loss: -0.9057 | ess: 0.9045
ConsoleLogger: epoch_loss = -0.9056745767593384
(cpu) Time = 2.72 sec.
###############################
# 'Fitter' executed in 4.229s #
###############################
#################################
# 'Pipeline' executed in 4.229s #
#################################
```

This output indicates the loss values at specified epochs during the training
process, providing insight into the model's performance over time.

For working on HPC systems,  a `startscript.sh` file is provided. This can be launched
by:
```
sbatch startscript.sh
```
In the startscript, `nodes` specifies the number of workers to be used for training.
You can efficiently scale your model training across multiple GPUs, enhancing
performance and reducing training time. This flexibility allows you to tackle
larger datasets and more complex models with ease.

This example demonstrates the flexibility of using the package to implement
scalar field theories in a simplified zero-dimensional setting. It can be
generalized to any dimension by changing the shape provided to the prior
distribution.

After training the model, one can draw samples using an attribute called
`posterior`.
To draw `n` samples from the trained distribution, use the following command:

```python
x = model.posterior.sample(n)
```

Note that the trained distribution is almost never identical to the target
distribution, which is specified by the action. To generate samples that are
correctly drawn from the target distribution, similar to Markov Chain Monte
Carlo (MCMC) simulations, one can employ a Metropolis accept/reject step and
discard some of the initial samples. To this end, you can use the following
command:

```python
x = model.mcmc.sample(n)
```

This command draws `n` samples from the trained distribution and applies a
Metropolis accept/reject step to ensure that the samples are correctly drawn.

<p align="center">
    <img src="docs/images/Normflow.png" alt="Block diagram for the method of normalizing flows" width="80%" />
</p>
<p align="center">
    Block diagram for the method of normalizing flows
</p>


The *TRAIN* and *GENERATE* blocks in the above figure depict the procedures for
training the model and generating samples/configurations. For more information
see [arXiv:2301.01504](https://arxiv.org/abs/2301.01504).

In summary, this package provides a robust and flexible framework for
implementing the method of normalizing flows as a generative model for lattice
field theory. With its intuitive design and support for scalar theories, you
can easily adapt it to various dimensions and leverage GPU acceleration for
efficient training. We encourage you to explore the features and capabilities
of the package, and we welcome contributions and feedback to help us improve
and expand its functionality.

| Created by Javad Komijani in 2021 \
| Copyright (C) 2021-24, Javad Komijani
