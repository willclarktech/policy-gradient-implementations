#!/usr/bin/env python3
import argparse

import actor_critic
import reinforce


def main(experiment: str) -> None:
    if experiment == "actor_critic":
        hyperparameters = actor_critic.default_hyperparameters
        agent = actor_critic.ActorCritic(hyperparameters)
        actor_critic.train(agent, hyperparameters)
    # elif experiment == "ddpg":
    #     pass
    elif experiment == "reinforce":
        hyperparameters = reinforce.default_hyperparameters
        agent = reinforce.Reinforce(hyperparameters)
        reinforce.train(agent, hyperparameters)
    # elif experiment == "sac":
    #     pass
    # elif experiment == "td3":
    #     pass
    else:
        raise ValueError(f"Experiment {experiment} not recognised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    main(args.experiment)
