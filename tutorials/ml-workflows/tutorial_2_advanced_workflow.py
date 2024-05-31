"""
In the first two tutorials we saw how to define simple sequential workflows by
means of the Pipeline object, which feds the outputs of the previous component
as inputs of the following one.
In this tutorial we show how to create more complex workflows, with
non-sequential data flows. Here, components can be arranges as an directed
acyclic graph (DAG). Under the DAG assumption, outputs of each block can be fed
as input potentially to any other component, granting great flexibility to the
experimenter.
The trade-off for improved flexibility is a change in the way we define
configuration files. From now on, it will only be possible to configure the
parameters used by the training script, but not its structure through the
Pipeline.

itwinai provides a wrapper of jsonarparse's ArgumentParser which supports
configuration files by default.

To run as usual:
>>> python my_script.py -d 20 --train-prop 0.7 --val-prop 0.2 --lr 1e-5

To reuse the parameters saved in a configuration file and override some
parameter (e.g., learning rate):
>>> python my_script.py --config advanced_tutorial_conf.yaml --lr 2e-3


"""
from typing import Any
from itwinai.parser import ArgumentParser
from itwinai.components import Predictor, monitor_exec

from basic_components import (
    MyDataGetter, MyDatasetSplitter, MyTrainer, MySaver
)


class MyEnsemblePredictor(Predictor):
    @monitor_exec
    def execute(self, dataset, model_ensemble) -> Any:
        """
        do some predictions with model on dataset...
        """
        return dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="itwinai advanced workflows tutorial")
    parser.add_argument(
        "--data-size", "-d", type=int, required=True,
        help="Dataset cardinality.")
    parser.add_argument(
        "--train-prop", type=float, required=True,
        help="Train split proportion.")
    parser.add_argument(
        "--val-prop", type=float, required=True,
        help="Validation split proportion.")
    parser.add_argument(
        "--lr", type=float, help="Training learning rate.")
    args = parser.parse_args()

    # Save parsed arguments to configuration file.
    # Previous configurations are overwritten, which is not good,
    # but the versioning of configuration files is out of the scope
    # of this tutorial.
    parser.save(
        args, "advanced_tutorial_conf.yaml", format='yaml', overwrite=True)

    # Define workflow components
    getter = MyDataGetter(data_size=args.data_size)
    splitter = MyDatasetSplitter(
        train_proportion=args.train_prop,
        validation_proportion=args.val_prop,
        test_proportion=1-args.train_prop-args.val_prop
    )
    trainer1 = MyTrainer(lr=args.lr)
    trainer2 = MyTrainer(lr=args.lr)
    saver = MySaver()
    predictor = MyEnsemblePredictor(model=None)

    # Define ML workflow
    dataset = getter.execute()
    train_spl, val_spl, test_spl = splitter.execute(dataset)
    _, _, _, trained_model1 = trainer1.execute(train_spl, val_spl, test_spl)
    _, _, _, trained_model2 = trainer2.execute(train_spl, val_spl, test_spl)
    _ = saver.execute(trained_model1)
    predictions = predictor.execute(test_spl, [trained_model1, trained_model2])
    print()
    print("Predictions: " + str(predictions))
