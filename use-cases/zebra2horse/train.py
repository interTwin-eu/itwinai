import argparse

from trainer import Zebra2HorseTrainer
from dataloader import Zebra2HorseDataLoader
from itwinai.backend.executors import LocalExecutor

if __name__ == "__main__":
    # Create CLI Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline", type=str)
    args = parser.parse_args()

    # Execute pipe
    executor = LocalExecutor(pipeline=args.pipeline, class_dict={
        "loader": Zebra2HorseDataLoader,
        "trainer": Zebra2HorseTrainer
    })
    executor.setup(None)
    executor.execute(None)
