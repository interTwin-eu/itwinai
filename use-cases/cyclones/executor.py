import logging
from os.path import join
from os import makedirs
from datetime import datetime
from typing import Tuple, Dict, Optional, Iterable

from lib.macros import PATCH_SIZE as patch_size, SHAPE as shape
from itwinai.components import Pipeline, BaseComponent


class CycloneExecutor(Pipeline):
    def __init__(
        self,
        run_name: str,
        steps: Iterable[BaseComponent],
        name: Optional[str] = None
    ):
        super().__init__(steps=steps, name=name)
        self.run_name = run_name

    def execute(
        self,
        root_dir,
        config: Optional[Dict] = None,
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        self.root_dir = root_dir
        print(f" Data will be stored at: {self.root_dir}")
        config = self.setup_config(config)
        super().execute(config=config)

    def setup_config(self, config: Optional[Dict] = None) -> Dict:
        config = config if config is not None else {}

        # Paths, Folders
        FORMATTED_DATETIME = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        MODEL_BACKUP_DIR = join(self.root_dir, "models/")
        EXPERIMENTS_DIR = join(self.root_dir, "experiments")
        RUN_DIR = join(EXPERIMENTS_DIR, self.run_name +
                       "_" + FORMATTED_DATETIME)
        SCALER_DIR = join(RUN_DIR, "scalers")
        TENSORBOARD_DIR = join(RUN_DIR, "tensorboard")
        CHECKPOINTS_DIR = join(RUN_DIR, "checkpoints")

        # Files
        LOG_FILE = join(RUN_DIR, "run.log")

        # Create folders
        makedirs(EXPERIMENTS_DIR, exist_ok=True)
        makedirs(RUN_DIR, exist_ok=True)
        makedirs(SCALER_DIR, exist_ok=True)
        makedirs(TENSORBOARD_DIR, exist_ok=True)
        makedirs(CHECKPOINTS_DIR, exist_ok=True)

        config = {
            "root_dir": self.root_dir,
            "experiment_dir": EXPERIMENTS_DIR,
            "run_dir": RUN_DIR,
            "scaler_dir": SCALER_DIR,
            "tensorboard_dir": TENSORBOARD_DIR,
            "checkpoints_dir": CHECKPOINTS_DIR,
            "backup_dir": MODEL_BACKUP_DIR,
            "log_file": LOG_FILE,
            "shape": shape,
            "patch_size": patch_size,
        }
        self.args = config

        # initialize logger
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s : %(message)s",
            level=logging.DEBUG,
            filename=LOG_FILE,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return config
