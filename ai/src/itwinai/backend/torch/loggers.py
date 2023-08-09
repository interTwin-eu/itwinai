import os
import wandb
import mlflow
import mlflow.keras

from ..components import Logger


class WanDBLogger(Logger):
    def __init__(self):
        pass

    def log(self, args):
        wandb.init(config={"bs": 12})


class MLFlowLogger(Logger):
    def __init__(self):
        pass

    def log(self, args):
        mlflow.pytorch.autolog()


class BaseLogger(Logger):
    def __init__(
            self,
            savedir: str = 'mllogs',
            create_new: bool = True
    ) -> None:
        super().__init__()
        self.savedir = savedir

        # From now on, very spaghetti...
        os.makedirs(self.savedir, exist_ok=True)
        if create_new:
            run_dirs = sorted(os.listdir(self.savedir))
            if len(run_dirs) == 0:
                self.run_id = 0
            else:
                self.run_id = int(run_dirs[-1]) + 1
            self.run_path = os.path.join(self.savedir, str(self.run_id))
            os.makedirs(self.run_path)
        else:
            # "Wait" for the process 0 to create the run folder...
            import time
            time.sleep(0.1)
            run_dirs = sorted(os.listdir(self.savedir))
            self.run_id = int(run_dirs[-1])
            self.run_path = os.path.join(self.savedir, str(self.run_id))

    def log(self, args):
        pass
