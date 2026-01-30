from networks.mlp import MLP
from networks.cnn import CNN

from configs.config import SingleNetworkConfig



NETWORK_MAPPING = {
    "mlp": MLP,
    "cnn": CNN,
}

def build_network(cfg: SingleNetworkConfig, obs_space, act_space):
    network = NETWORK_MAPPING[cfg.network_type]
    return network(
        cfg = cfg, 
        obs_space = obs_space,
        act_space = act_space
    )