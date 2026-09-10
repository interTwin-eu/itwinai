class BaseBackend:
    def __init__(self, config: dict):
        self.config = config

    def upload(self, model_dir):
        raise NotImplementedError
