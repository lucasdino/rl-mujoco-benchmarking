from algorithms.sac import SoftActorCritic
from algorithms.ddqn import DDQN

ALGO_MAP = {
    "sac": SoftActorCritic,
    "ddqn": DDQN
}


def get_algorithm(algorithm_name):
    return ALGO_MAP[algorithm_name]