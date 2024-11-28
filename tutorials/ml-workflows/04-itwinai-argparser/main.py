# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Any

from basic_components import MyDataGetter, MyDatasetSplitter, MySaver, MyTrainer

from itwinai.components import Predictor, monitor_exec
from itwinai.parser import ArgumentParser


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
        "--data-size", "-d", type=int, required=True, help="Dataset cardinality."
    )
    parser.add_argument(
        "--train-prop", type=float, required=True, help="Train split proportion."
    )
    parser.add_argument(
        "--val-prop", type=float, required=True, help="Validation split proportion."
    )
    parser.add_argument("--lr", type=float, help="Training learning rate.")
    args = parser.parse_args()

    # Save parsed arguments to configuration file.
    # Previous configurations are overwritten, which is not good,
    # but the versioning of configuration files is out of the scope
    # of this tutorial.
    parser.save(args, "advanced_tutorial_conf.yaml", format="yaml", overwrite=True)

    # Define workflow components
    getter = MyDataGetter(data_size=args.data_size)
    splitter = MyDatasetSplitter(
        train_proportion=args.train_prop,
        validation_proportion=args.val_prop,
        test_proportion=1 - args.train_prop - args.val_prop,
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
