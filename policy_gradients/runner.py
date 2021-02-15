import json
from pprint import pprint
import signal
import sys
from tempfile import mkdtemp
from typing import Any, Callable, Dict, Optional, Tuple

from policy_gradients.core import (
    Algorithm,
    BaseAgent,
    Hyperparameters,
    train,
    TrainOptions,
)
from policy_gradients.utils import set_seed

from policy_gradients.actor_critic import algorithm as actor_critic
from policy_gradients.baseline import algorithm as baseline
from policy_gradients.ddpg import algorithm as ddpg
from policy_gradients.human import algorithm as human
from policy_gradients.ppo import algorithm as ppo
from policy_gradients.reinforce import algorithm as reinforce
from policy_gradients.sac import algorithm as sac
from policy_gradients.td3 import algorithm as td3

Options = Dict[str, Any]

algorithms: Dict[str, Algorithm] = {
    "actor_critic": actor_critic,
    "baseline": baseline,
    "ddpg": ddpg,
    "human": human,
    "ppo": ppo,
    "reinforce": reinforce,
    "sac": sac,
    "td3": td3,
}


def create_interrupt_handler(
    agent: BaseAgent, save_dir: str
) -> Callable[[int, Any], None]:
    def handle_interrupt(signal_number: int, frame: Any) -> None:
        # pylint: disable=unused-argument
        print("Training interrupted")
        answer = None

        while answer not in ["", "n", "no", "y", "yes"]:
            answer = input("Do you want to save the model? [y/N] ").strip().lower()

        if answer in ["y", "yes"]:
            print(f"Saving model to {save_dir}...")
            agent.save(save_dir)
            print("Successfully saved model")

        sys.exit(0)

    return handle_interrupt


def maybe_set_seed(options: Options) -> None:
    seed = options.get("seed")
    if seed is not None:
        set_seed(seed)


def get_algorithm(options: Options) -> Tuple[str, Algorithm]:
    algorithm_name = options.get("algorithm") or ""
    algorithm = algorithms[algorithm_name]
    if algorithm is None:
        raise ValueError(f"Experiment {algorithm_name or '(none)'} not recognized")
    return algorithm_name, algorithm


def get_env_name(options: Options, defaults: Options) -> str:
    env_name = options.get("env_name") or defaults.get("env_name")
    if env_name is None:
        raise ValueError("No environment specified")
    return env_name


def load_hyperparameter_args(
    algorithm_name: str, env_name: str, load_dir: Optional[str]
) -> Options:
    return (
        json.loads(
            open(
                f"{load_dir}/{algorithm_name}_{env_name}/hyperparameters.json", "r"
            ).read()
        )
        if load_dir is not None
        else {}
    )


def gather_hyperparameter_args(
    default_hyperparameter_args: Options,
    loaded_hyperparameter_args: Options,
    options: Options,
) -> Options:
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
    return hyperparameter_args


def run(options: Dict[str, Any]) -> BaseAgent:
    maybe_set_seed(options)
    algorithm_name, algorithm = get_algorithm(options)

    should_eval = options.pop("eval", False)
    should_render = options.pop("render", False)
    load_dir = options.pop("load_dir", None)
    save_dir = options.pop("save_dir", None)
    tmp_dir = options.pop("tmp_dir", mkdtemp())
    print(f"Using temporary directory {tmp_dir}")

    default_hyperparameter_args = algorithm.default_hyperparameters()
    env_name = get_env_name(options, default_hyperparameter_args)

    loaded_hyperparameter_args = load_hyperparameter_args(
        algorithm_name, env_name, load_dir
    )
    hyperparameter_args = gather_hyperparameter_args(
        default_hyperparameter_args, loaded_hyperparameter_args, options
    )
    hyperparameters = Hyperparameters(**hyperparameter_args)
    agent = algorithm.Agent(hyperparameters)

    if load_dir is not None:
        print(f"Loading model from {load_dir}...")
        agent.load(load_dir)
        print("Successfully loaded model")

    if save_dir is not None:
        signal.signal(signal.SIGINT, create_interrupt_handler(agent, save_dir))

    if should_eval:
        agent.eval()

    print(f"Algorithm: {algorithm_name}")
    print("Hyperparameters:")
    pprint(hyperparameter_args)

    train_options = TrainOptions(should_eval, should_render, tmp_dir)
    print("Starting training...")
    train(
        agent,
        hyperparameters,
        algorithm.run_episode,
        train_options,
    )
    print("Finished training")

    if save_dir is not None:
        print(f"Saving model to {save_dir}...")
        agent.save(save_dir)
        print("Successfully saved model")

    return agent
