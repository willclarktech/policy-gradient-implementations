import numpy as np  # type: ignore

from core import Hyperparameters
from utils import plot_returns

from actor_critic.agent import Agent


def run_episode(agent: Agent, hyperparameters: Hyperparameters) -> float:
    env = hyperparameters.env
    observation = env.reset()
    done = False
    ret = 0.0

    while not done:
        action, log_probability = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.update(observation, log_probability, reward, done, observation_)
        observation = observation_

    return ret
