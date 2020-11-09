#!/usr/bin/env python3
import argparse
from typing import Optional

from core import train
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


def main(experiment: str, seed: Optional[int]) -> None:
    if seed is not None:
        set_seed(seed)

    algorithm = algorithms[experiment]
    if algorithm is None:
        raise ValueError(f"Experiment {experiment} not recognized")

    hyperparameters = algorithm.default_hyperparameters(seed)  # type: ignore
    agent = algorithm.Agent(hyperparameters)  # type: ignore
    train(agent, hyperparameters, algorithm.run_episode)  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment", help=f"Choose from: {', '.join(algorithms.keys())}"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.experiment, args.seed)
