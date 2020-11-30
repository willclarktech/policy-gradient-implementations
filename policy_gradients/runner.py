from pprint import pprint
from typing import Any, Dict

from policy_gradients.core import BaseAgent, Hyperparameters, train
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


def run(options: Dict[str, Any]) -> BaseAgent:
    if "seed" in options and options["seed"] is not None:
        set_seed(options["seed"])

    load_dir = options.pop("load_dir", None)
    save_dir = options.pop("save_dir", None)
    should_render = options.pop("render", False)

    algorithm_name = options["algorithm"]
    algorithm = algorithms[algorithm_name]
    if algorithm is None:
        raise ValueError(f"Experiment {algorithm_name} not recognized")

    hyperparameter_args = algorithm.default_hyperparameters()  # type: ignore
    for key in options:
        if options[key] is not None:
            hyperparameter_args[key] = options[key]

    hyperparameters = Hyperparameters(**hyperparameter_args)
    agent = algorithm.Agent(hyperparameters)  # type: ignore

    if load_dir is not None:
        print(f"Loading model from {load_dir}...")
        agent.load(load_dir)
        print("Successfully loaded model")

    print(f"Algorithm: {algorithm_name}")
    print("Hyperparameters:")
    pprint(hyperparameter_args)

    print("Starting training...")
    train(agent, hyperparameters, algorithm.run_episode, save_dir=save_dir, should_render=should_render)  # type: ignore
    print("Finished training")

    if save_dir is not None:
        agent.save(save_dir)

    return agent
