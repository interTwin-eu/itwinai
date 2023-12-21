from ..distributed import DistributedStrategy


class TFDistributedStrategy(DistributedStrategy):
    """Abstract class to define the distributed backend methods for
    TensorFlow models.
    """
