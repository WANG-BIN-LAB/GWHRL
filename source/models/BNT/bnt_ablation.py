from omegaconf import DictConfig

from .bnt import BrainNetworkTransformer
from .components.BrainSubNetAblation import (
    RandomSubnetGraphFlattenTokenizer,
    SequentialEqualSubnetGraphFlattenTokenizer,
)


DEFAULT_SUBNET_PATH = (
    "F:/TYUT/Projects/BrainNetworkTransformer-main/"
    "BrainNetworkTransformer-main/figure/cc200.csv"
)


def _model_value(config: DictConfig, name: str, default):
    if not hasattr(config, "model") or not hasattr(config.model, name):
        return default
    return getattr(config.model, name)


class RandomSubnetBrainNetworkTransformer(BrainNetworkTransformer):
    """
    Ablation model 1:
    Randomly divide 200 nodes into 8 subnets while preserving the biological
    subnet sizes, then reuse all downstream operations from the main model.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        subnet_path = _model_value(config, "subnet_path", DEFAULT_SUBNET_PATH)
        num_nodes = _model_value(config, "num_nodes", 200)
        embed_dim = _model_value(config, "spd_output_vector_dim", 128)
        random_seed = _model_value(config, "random_subnet_seed", 3)

        self.converter = RandomSubnetGraphFlattenTokenizer(
            subnet_path=subnet_path,
            num_nodes=num_nodes,
            embed_dim=embed_dim,
            seed=random_seed,
        )


class SequentialEqualSubnetBrainNetworkTransformer(BrainNetworkTransformer):
    """
    Ablation model 2:
    Sequentially divide 200 nodes into 8 equal subnets of 25 nodes each, then
    reuse all downstream operations from the main model.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        num_nodes = _model_value(config, "num_nodes", 200)
        embed_dim = _model_value(config, "spd_output_vector_dim", 128)
        num_subnets = _model_value(config, "num_subnets", 8)

        self.converter = SequentialEqualSubnetGraphFlattenTokenizer(
            num_nodes=num_nodes,
            embed_dim=embed_dim,
            num_subnets=num_subnets,
        )
