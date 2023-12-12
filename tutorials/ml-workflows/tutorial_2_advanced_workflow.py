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
>>> python my_script.py --config my_config_file.yaml --lr 2e-3

"""
from itwinai.parser import ArgumentParser

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
    # parser.save(args, "test_conf2.yaml", format='yaml')
