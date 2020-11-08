#!/usr/bin/env python3
import argparse

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


def main(experiment: str) -> None:
    algorithm = algorithms[experiment]
    if algorithm is None:
        raise ValueError(f"Experiment {experiment} not recognised")

    hyperparameters = algorithm.default_hyperparameters
    agent = algorithm.Agent(hyperparameters)
    algorithm.train(agent, hyperparameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment", help=f"Choose from: {', '.join(algorithms.keys())}"
    )
    args = parser.parse_args()
    main(args.experiment)
