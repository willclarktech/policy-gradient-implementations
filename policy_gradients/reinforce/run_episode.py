import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from policy_gradients.core import Hyperparameters
from policy_gradients.utils import plot_returns

from policy_gradients.reinforce.agent import Agent


def run_episode(agent: Agent, hyperparameters: Hyperparameters) -> float:
    env = hyperparameters.env
    observation = env.reset()
    agent.reset()
    done = False
    ret = 0.0

    while not done:
        action, log_probability = agent.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        ret += reward
        agent.remember(log_probability, reward)
        observation = observation_

    return ret
