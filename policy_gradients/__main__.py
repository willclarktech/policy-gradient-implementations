#!/usr/bin/env python3
import argparse

import actor_critic
import ddpg
import reinforce
import td3


def main(experiment: str) -> None:
    if experiment == "actor_critic":
        hyperparameters = actor_critic.default_hyperparameters
        agent = actor_critic.ActorCritic(hyperparameters)
        actor_critic.train(agent, hyperparameters)
    elif experiment == "ddpg":
        hyperparameters = ddpg.default_hyperparameters
        agent = ddpg.DDPG(hyperparameters)
        ddpg.train(agent, hyperparameters)
    elif experiment == "reinforce":
        hyperparameters = reinforce.default_hyperparameters
        agent = reinforce.Reinforce(hyperparameters)
        reinforce.train(agent, hyperparameters)
    # elif experiment == "sac":
    #     pass
    elif experiment == "td3":
        hyperparameters = td3.default_hyperparameters
        agent = td3.TD3(hyperparameters)
        td3.train(agent, hyperparameters)
    else:
        raise ValueError(f"Experiment {experiment} not recognised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    main(args.experiment)
