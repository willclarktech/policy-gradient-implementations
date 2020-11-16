from pprint import pprint
from typing import Any, Dict

from policy_gradients.core import Hyperparameters, train
from policy_gradients.parser import create_parser
from policy_gradients.utils import set_seed

import policy_gradients.actor_critic as actor_critic
import policy_gradients.ddpg as ddpg
import policy_gradients.reinforce as reinforce
import policy_gradients.sac as sac
import policy_gradients.td3 as td3

algorithms = {
    "actor_critic": actor_critic,
    "ddpg": ddpg,
    "reinforce": reinforce,
    "sac": sac,
    "td3": td3,
}


def run(cli_args: Dict[str, Any]) -> None:
    if hasattr(cli_args, "seed") and cli_args["seed"] is not None:
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
