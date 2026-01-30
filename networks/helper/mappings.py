import torch.nn as nn
from networks.helper.noisylinear import NoisyLinear


ACTIV_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "identity": nn.Identity

}

LINEAR_LAYER_MAP = {
    "linear": nn.Linear,
    "noisy_linear": NoisyLinear,
    "Linear": nn.Linear,
    "NoisyLinear": NoisyLinear
}