import logging

from os.path import join, exists
from os import listdir, makedirs
from itwinai.backend.components import Executor
from datetime import datetime
from lib.macros import PATCH_SIZE as patch_size, SHAPE as shape

class CycloneExecutor(Executor):
    def __init__(
            self,
            run_name:str
    ):
        self.run_name = run_name

    def execute(self, pipeline):
        args = None
        for executable in pipeline:
            args = executable.execute(args)

    def setup(self, args):
        pipeline, root_dir = args

        # Paths, Folders
        FORMATTED_DATETIME = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        MODEL_BACKUP_DIR = join(root_dir, 'models/')
        EXPERIMENTS_DIR = join(root_dir, 'experiments')
        RUN_DIR = join(EXPERIMENTS_DIR, run_name + '_' + FORMATTED_DATETIME)
        SCALER_DIR = join(RUN_DIR, 'scalers')
        TENSORBOARD_DIR = join(RUN_DIR, 'tensorboard')
        CHECKPOINTS_DIR = join(RUN_DIR, 'checkpoints')

        # Files
        LOG_FILE = join(RUN_DIR, 'run.log')

        # Create folders
        makedirs(EXPERIMENTS_DIR, exist_ok=True)
        makedirs(RUN_DIR, exist_ok=True)
        makedirs(SCALER_DIR, exist_ok=True)
        makedirs(TENSORBOARD_DIR, exist_ok=True)
        makedirs(CHECKPOINTS_DIR, exist_ok=True)

        self.args = {
            'root_dir': root_dir,
            'experiment_dir': EXPERIMENTS_DIR,
            'run_dir': RUN_DIR,
            'scaler_dir': SCALER_DIR,
            'tensorboard_dir': TENSORBOARD_DIR,
            'checkpoints_dir': CHECKPOINTS_DIR,
            'backup_dir': MODEL_BACKUP_DIR,
            'log_file': LOG_FILE,
            'shape': shape,
            'patch_size': patch_size,
        }

        # initialize logger
        logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", level=logging.DEBUG, filename=LOG_FILE,
                            datefmt='%Y-%m-%d %H:%M:%S')

        for executable in pipeline:
            self.args = executable.setup(self.args)