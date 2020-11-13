from pprint import pprint
from typing import Any, Dict

from core import Hyperparameters, train
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


def main(cli_args: Dict[str, Any]) -> None:
    if cli_args["seed"] is not None:
        set_seed(cli_args["seed"])

    algorithm_name = cli_args.pop("algorithm")
    algorithm = algorithms[algorithm_name]
    if algorithm is None:
        raise ValueError(f"Experiment {algorithm_name} not recognized")

    hyperparameter_args = algorithm.default_hyperparameters()  # type: ignore
    for key in cli_args:
        if cli_args[key] is not None:
            hyperparameter_args[key] = cli_args[key]

    hyperparameters = Hyperparameters(**hyperparameter_args)
    agent = algorithm.Agent(hyperparameters)  # type: ignore

    print(f"Algorithm: {algorithm_name}")
    print("Hyperparameters:")
    pprint(hyperparameter_args)

    train(agent, hyperparameters, algorithm.run_episode)  # type: ignore


if __name__ == "__main__":
    parser = create_parser(algorithms.keys())
    args = parser.parse_args()
    main(vars(args))
