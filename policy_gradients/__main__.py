from typing import Optional

from core import train
from parser import create_parser
from utils import set_seed

import actor_critic
import ddpg
import reinforce
import sac
import td3

algorithms = {
    "actor_critic": actor_critic,
    "ddpg": ddpg,
    "reinforce": reinforce,
    "sac": sac,
    "td3": td3,
}


def main(args) -> None:
    if args.seed is not None:
        set_seed(args.seed)

    algorithm = algorithms[args.experiment]
    if algorithm is None:
        raise ValueError(f"Experiment {args.experiment} not recognized")

    hyperparameters = algorithm.default_hyperparameters(args.seed)  # type: ignore
    agent = algorithm.Agent(hyperparameters)  # type: ignore
    train(agent, hyperparameters, algorithm.run_episode)  # type: ignore


if __name__ == "__main__":
    parser = create_parser(algorithms.keys())
    args = parser.parse_args()
    main(args)
