import json
from pprint import pprint
from typing import Any, Dict

from policy_gradients.core import Algorithm, BaseAgent, Hyperparameters, train
from policy_gradients.utils import set_seed

from policy_gradients.actor_critic import algorithm as actor_critic
from policy_gradients.baseline import algorithm as baseline
from policy_gradients.ddpg import algorithm as ddpg
from policy_gradients.human import algorithm as human
from policy_gradients.reinforce import algorithm as reinforce
from policy_gradients.sac import algorithm as sac
from policy_gradients.td3 import algorithm as td3

algorithms: Dict[str, Algorithm] = {
    "actor_critic": actor_critic,
    "baseline": baseline,
    "ddpg": ddpg,
    "human": human,
    "reinforce": reinforce,
    "sac": sac,
    "td3": td3,
}


def run(options: Dict[str, Any]) -> BaseAgent:
    if "seed" in options and options["seed"] is not None:
        set_seed(options["seed"])

    algorithm_name = options["algorithm"]
    algorithm = algorithms[algorithm_name]
    if algorithm is None:
        raise ValueError(f"Experiment {algorithm_name} not recognized")

    load_dir = options.pop("load_dir", None)
    save_dir = options.pop("save_dir", None)
    should_eval = options.pop("eval", False)
    should_render = options.pop("render", False)

    default_hyperparameter_args = algorithm.default_hyperparameters()
    env_name = options.get("env_name") or default_hyperparameter_args["env_name"]
    loaded_hyperparameter_args = (
        json.loads(
            open(
                f"{load_dir}/{algorithm_name}_{env_name}/hyperparameters.json", "r"
            ).read()
        )
        if load_dir is not None
        else {}
    )
    filtered_loaded_hyperparameter_args = {
        k: v
        for k, v in loaded_hyperparameter_args.items()
        if k in default_hyperparameter_args and v is not None
    }
    filtered_options = {
        k: v
        for k, v in options.items()
        if k in default_hyperparameter_args and v is not None
    }
    hyperparameter_args = {
        **default_hyperparameter_args,
        **filtered_loaded_hyperparameter_args,
        **filtered_options,
    }

    # for key in hyperparameter_args:
    #     if loaded_hyperparameter_args[key] is not None:
    #         hyperparameter_args[key] = loaded_hyperparameter_args[key]
    # for key in options:
    #     if options[key] is not None:
    #         hyperparameter_args[key] = options[key]

    hyperparameters = Hyperparameters(**hyperparameter_args)
    agent = algorithm.Agent(hyperparameters)

    if load_dir is not None:
        print(f"Loading model from {load_dir}...")
        agent.load(load_dir)
        print("Successfully loaded model")

    if should_eval:
        agent.eval()

    print(f"Algorithm: {algorithm_name}")
    print("Hyperparameters:")
    pprint(hyperparameter_args)

    print("Starting training...")
    train(
        agent,
        hyperparameters,
        algorithm.run_episode,
        should_render=should_render,
        should_eval=should_eval,
    )
    print("Finished training")

    if save_dir is not None:
        agent.save(save_dir)

    return agent
