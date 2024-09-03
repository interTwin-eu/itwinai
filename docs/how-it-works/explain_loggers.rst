===============
Logging data with Itwinai
-------------

One key aspect of understanding the intricacies of your ML experiment lies in the descriptive data around parameters, metrics, and outputs.
This information allows us to compare different models and reproduce previous results.


In order to facilitate this, Itwinai provides a `logger` wrapper which provides users with a simple, standardised method to extract descriptive information about their ML training run regardless of which supported logger is used.
The supported loggers, also shown in figure :numref:`logger_fig`, are listed in :doc:`this table <../../src/loggers.py>`

.. _logger fig:
.. image::  workflows/figures/logger_fig.png
    :alt: Diagram of supported loggers in Itwinai `logger` wrapper
    :align: center
    :scale: 12%


Getting started with loggers
----------------------

Itwinai provides a wrapper that allows the user to call a logger in a standardised manner regardless of which logger is used on the backend.

`log` is used to log data of interest from training runs. See `itwinai.loggers` for a list of objects that can be logged

The itwinai logger also contains a dashboard for logged data visualisation.

The `log_freq` parameter allows the user to determine at which batch interval the logger is called.
When an integer is given for this parameter, the logger will log when that batch's batch id is a multiple of the given integer.
Thus, should `log_freq = 5`, the first batch (batch_id 0) is logged, after which the 11th batch (batch_id 10) is logged, after which the 20th batch is logged and so

`log_freq` can also be set to the values `epoch` or `batch`.
When set to `epoch`, the logger only logs the epoch.
The logger only logs if `batch_idx` is not passed as parameter to `log`
When set to `batch`, every batch is logged 



Further references
-------------------
- `WandB documentation <https://docs.wandb.ai/ref/python/watch>`_
- `Tensorboard log documentation <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html>`_
- `MLFlow logger documentation <https://mlflow.org/docs/latest/tracking/tracking-api.html#manual-logging>`_