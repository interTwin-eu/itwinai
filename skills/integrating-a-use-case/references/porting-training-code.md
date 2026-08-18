# Porting the training code

## How a pipeline moves data

`itwinai.pipeline.Pipeline` runs its steps in order and passes each step's return value into the
next step's `execute()` as positional arguments. That is the whole contract. A pipeline that
ends in a trainer therefore needs its data-producing step to return exactly what the trainer's
`execute()` accepts.

`TorchTrainer.execute` takes `(train_dataset, validation_dataset=None, test_dataset=None)` and
returns `(train, validation, test, model)`. Read the signature in
`itwinai/torch/trainer.py` rather than trusting this line.

## Branch A - components only (the default)

No trainer subclass. Write a `DataGetter` that returns the three datasets, and let the stock
`TorchTrainer` do the rest.

```python
from typing import Tuple

from torch.utils.data import Dataset, Subset

from itwinai.components import DataGetter, monitor_exec


class MyDataGetter(DataGetter):
    def __init__(self, data_root: str, max_train_size: int | None = None) -> None:
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.data_root = data_root
        self.max_train_size = max_train_size

    @monitor_exec
    def execute(self) -> Tuple[Dataset, Dataset, None]:
        train, validation = build_my_datasets(self.data_root)
        if self.max_train_size:
            train = Subset(train, range(self.max_train_size))
        return train, validation, None
```

Three things are mandatory and easy to miss:

- `@monitor_exec` on `execute`. Without it the component is not tracked.
- `self.save_parameters(**self.locals2params(locals()))` in `__init__`. Without it the
  component's configuration is not serialised into the run record.
- Return the full 3-tuple, using `None` for splits you do not have.

`max_train_size` is not decoration. A truncation knob makes Gate 3 and Gate 4 cheap, and the
MNIST plugin ships exactly this pair of arguments for that reason. Add one.

Other component base classes in `itwinai/components.py` - `DataProcessor`, `DataSplitter`,
`Predictor`, `Saver`, `Adapter` - follow the same rules and exist for the same purpose. Use
`DataSplitter` when the source is one undifferentiated dataset that needs splitting.

## Branch B - custom trainer

Subclass `itwinai.torch.trainer.TorchTrainer` and override the smallest set of methods that
covers the difference. Read the method you intend to override before overriding it; several
carry an explicit comment marking them as user-overridable.

| Override | When |
|---|---|
| `create_model_loss_optimizer` | The loss is not in `TrainingConfiguration`'s allowed set, or the optimizer or scheduler is constructed in a non-standard way, or the model needs post-construction setup. **The most common override.** |
| `create_dataloaders` | Custom collation, samplers, or a non-`Dataset` source. |
| `train_step` | One batch is handled unusually - custom backward, multiple losses. |
| `train_epoch` | The epoch itself is unusual - multiple optimizers, alternating updates, per-batch normalisation. |
| `validation_step` / `validation_epoch` | Validation differs structurally from training. |
| `compute_metrics` | Metrics need inputs beyond `(prediction, target)`. |
| `train` | Rarely. The whole loop is unusual, e.g. physics-residual training with no dataloader. Overriding this discards checkpointing, profiling and reporting unless you re-implement them. Prefer a narrower override. |
| `execute` | Almost never. Only to set up state before the distributed environment is initialised. |

Whatever you override, call `super()` where the base method does bookkeeping you still want, and
keep using `self.strategy`, `self.device`, `self.log(...)` and `self.ray_report(...)` rather than
raw torch equivalents. Bypassing them is what breaks distributed runs and empty-report bugs
later.

### Custom hyperparameters

`TrainingConfiguration` is a pydantic model that permits extra fields, so unrecognised keys in
`config:` survive and are reachable as `self.config.<key>`. That is enough for most cases.

Subclass it when you want the extra fields validated and documented:

```python
from itwinai.torch.config import TrainingConfiguration


class MyTrainingConfiguration(TrainingConfiguration):
    modes: int = 12
    width: int = 32
```

Read `itwinai/torch/config.py` for the fields that already exist - notably the allowed `loss`
and `optimizer` values - before adding your own. Duplicating a field that is already there under
a different name is a common and confusing mistake.

## Do not shadow itwinai names

One published plugin defines its own class called `TorchTrainer`. Any reader, and any future
`_target_`, now has to disambiguate between it and `itwinai.torch.trainer.TorchTrainer`. Name
your classes after the science: `FNOTrainer`, `RNNDistributedTrainer`, `PulsarTrainer`.

## Preserve behaviour

Porting is a structural change. Keep the original script until Gate 3 passes, run both for one
epoch on the same seed and data, and compare the loss. If they differ materially, you changed
the science by accident. Fix that before moving on, and do not "improve" the model while
porting - it makes the discrepancy impossible to attribute.
