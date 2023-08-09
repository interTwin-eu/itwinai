import argparse

from trainer import Zebra2HorseTrainer
from dataloader import Zebra2HorseDataLoader
from itwinai.backend.executors import LocalExecutor, RayExecutor
from ray import tune


if __name__ == "__main__":
    # Create CLI Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline", type=str)
    args = parser.parse_args()

    # Execute pipe
    executor = RayExecutor(pipeline=args.pipeline, class_dict={
        "loader": Zebra2HorseDataLoader,
        "trainer": Zebra2HorseTrainer
    }, param_space={
        "trainer": {
            "init_args": {
                "batch_size": tune.randint(32, 512),
            }
        }
    })
    executor.setup(None)
    executor.execute(None)
